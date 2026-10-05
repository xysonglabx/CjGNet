# -*- coding: utf-8 -*-
"""
Enhanced HiGNN with PharmHGT-inspired three-level hierarchy
Integrates atom-pharmacophore-molecule multi-scale modeling
With attention weight extraction capabilities
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Parameter, Bilinear

from torch_scatter import scatter, scatter_add
from torch_geometric.nn import global_add_pool, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster


# ---------------------------------------
# Multi-View Attention (PharmHGT-inspired) - Enhanced with attention extraction
# ---------------------------------------
class MultiViewAttention(nn.Module):
    """
    Multi-view attention mechanism for heterogeneous graph nodes
    Handles atom-pharmacophore-molecule interactions with type-aware attention
    Enhanced with attention weight extraction capabilities
    """

    def __init__(self, hidden_channels, num_heads=8, num_node_types=3, num_edge_types=5, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout

        # Store attention weights for extraction
        self._last_attention_weights = None

        # Node type embeddings (atom, pharmacophore, molecule)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_channels)

        # Edge type embeddings (chemical bond, BRICS, junction, etc.)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_channels)

        # Multi-head attention projections
        self.W_q = Linear(hidden_channels, hidden_channels)
        self.W_k = Linear(hidden_channels, hidden_channels)
        self.W_v = Linear(hidden_channels, hidden_channels)

        # Meta-relation weights for different interaction types
        self.meta_relation_weights = nn.ParameterDict({
            'atom_to_pharm': Parameter(torch.Tensor(hidden_channels, hidden_channels)),
            'pharm_to_mol': Parameter(torch.Tensor(hidden_channels, hidden_channels)),
            'atom_to_mol': Parameter(torch.Tensor(hidden_channels, hidden_channels)),
            'pharm_to_pharm': Parameter(torch.Tensor(hidden_channels, hidden_channels)),
        })

        # Output projection
        self.out_proj = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.meta_relation_weights.values():
            glorot(param)
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        self.W_v.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x, edge_index, node_types, edge_types, relation_type):
        # Add node type information
        type_enhanced = x + self.node_type_embedding(node_types)

        # Compute Q, K, V
        Q = self.W_q(type_enhanced)
        K = self.W_k(type_enhanced)
        V = self.W_v(type_enhanced)

        # Apply relation-specific transformation
        if relation_type in self.meta_relation_weights:
            Q = torch.matmul(Q, self.meta_relation_weights[relation_type])

        # Reshape for multi-head attention
        batch_size = x.size(0)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Store attention weights for extraction
        self._last_attention_weights = attn_probs.detach()

        # Apply attention
        out = torch.matmul(attn_probs, V)
        out = out.view(batch_size, self.hidden_channels)

        return self.out_proj(out)


# ---------------------------------------
# Junction View Builder
# ---------------------------------------
class JunctionViewBuilder(nn.Module):
    """
    Constructs junction-level view connecting atoms, pharmacophores, and molecules
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Junction edge generators
        self.atom_pharm_edge = Linear(2 * hidden_channels, 1)
        self.pharm_mol_edge = Linear(2 * hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_pharm_edge.reset_parameters()
        self.pharm_mol_edge.reset_parameters()

    def forward(self, atom_features, pharm_features, atom_to_pharm_map):
        """
        Build junction edges between different hierarchical levels
        """
        junction_edges = []
        junction_weights = []

        # Build atom-pharmacophore connections
        if atom_to_pharm_map is not None and atom_to_pharm_map.numel() > 0:
            for edge in atom_to_pharm_map:
                atom_idx, pharm_idx = edge[0].item(), edge[1].item()
                if atom_idx < atom_features.size(0) and pharm_idx < pharm_features.size(0):
                    edge_feat = torch.cat([atom_features[atom_idx], pharm_features[pharm_idx]], dim=-1)
                    weight = torch.sigmoid(self.atom_pharm_edge(edge_feat))
                    junction_edges.append([atom_idx, pharm_idx])
                    junction_weights.append(weight)

        if len(junction_edges) > 0:
            junction_edges = torch.tensor(junction_edges, device=atom_features.device).t()
            junction_weights = torch.cat(junction_weights)
        else:
            junction_edges = torch.zeros((2, 0), dtype=torch.long, device=atom_features.device)
            junction_weights = torch.zeros(0, device=atom_features.device)

        return junction_edges, junction_weights


# ---------------------------------------
# Three-Level Hierarchical Fusion
# ---------------------------------------
class TriLevelHierarchicalFusion(nn.Module):
    """
    Fuses information across three hierarchical levels: atom, pharmacophore, molecule
    """

    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Stage 1: Atom to Pharmacophore fusion
        self.atom_to_pharm_attn = MultiViewAttention(
            hidden_channels, num_heads=num_heads, dropout=dropout
        )

        # Stage 2: Pharmacophore to Molecule fusion (preserving HiGNN logic)
        self.pharm_to_mol_attn = GATConv(
            hidden_channels, hidden_channels, heads=num_heads,
            dropout=dropout, concat=False
        )

        # Stage 3: Cross-scale fusion using junction view
        self.cross_scale_fusion = MultiViewAttention(
            hidden_channels, num_heads=num_heads, dropout=dropout
        )

        # Final projection
        self.final_proj = Sequential(
            Linear(3 * hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(2 * hidden_channels, hidden_channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_to_pharm_attn.reset_parameters()
        self.pharm_to_mol_attn.reset_parameters()
        self.cross_scale_fusion.reset_parameters()
        for module in self.final_proj:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, atom_feat, pharm_feat, mol_feat,
                atom_to_pharm_edges, pharm_to_mol_edges, junction_graph):
        """
        Hierarchical fusion following the principle: atom → pharmacophore → molecule
        """
        # Check if pharmacophores are empty
        has_pharm = pharm_feat.size(0) > 0

        if has_pharm:
            # Stage 1: Bottom-up aggregation (atom → pharmacophore)
            enhanced_pharm = self.atom_to_pharm_attn(
                pharm_feat, atom_to_pharm_edges,
                node_types=torch.ones(pharm_feat.size(0), dtype=torch.long, device=pharm_feat.device),
                edge_types=torch.zeros(atom_to_pharm_edges.size(1), dtype=torch.long, device=pharm_feat.device),
                relation_type='atom_to_pharm'
            )
            enhanced_pharm = enhanced_pharm + pharm_feat  # Residual connection

            # Stage 2: Continue bottom-up (pharmacophore → molecule)
            enhanced_mol = self.pharm_to_mol_attn(
                (enhanced_pharm, mol_feat), pharm_to_mol_edges
            )
            enhanced_mol = enhanced_mol + mol_feat  # Residual connection

            # Stage 3: Cross-scale interaction via junction view
            cross_scale_feat = self.cross_scale_fusion(
                torch.cat([atom_feat, enhanced_pharm, enhanced_mol], dim=0),
                junction_graph['edges'],
                junction_graph['node_types'],
                junction_graph['edge_types'],
                relation_type='cross_scale'
            )

            # Extract molecule-level features from cross-scale
            mol_cross_feat = cross_scale_feat[-mol_feat.size(0):]

            # Aggregate pharm features
            aggregated_pharm = global_add_pool(enhanced_pharm, pharm_to_mol_edges[1])

            # Final fusion combining all scales
            final_feat = torch.cat([
                enhanced_mol,  # Global molecular features
                mol_cross_feat,  # Cross-scale interactions
                aggregated_pharm  # Aggregated pharm features
            ], dim=-1)
        else:
            # No pharmacophores - use simplified fusion with just atoms and molecule
            # Use mol_feat directly without pharmacophore enhancement
            enhanced_mol = mol_feat

            # Create a simplified cross-scale feature using only atoms and molecule
            if atom_feat.size(0) > 0:
                # Simple aggregation from atoms to molecule
                atom_batch = torch.zeros(atom_feat.size(0), dtype=torch.long, device=atom_feat.device)
                aggregated_atoms = global_add_pool(atom_feat, atom_batch)

                # Final fusion with only atom and molecule features (no pharmacophores)
                # Pad with zeros to maintain expected dimension
                pharm_placeholder = torch.zeros(mol_feat.size(0), self.hidden_channels, device=mol_feat.device)

                final_feat = torch.cat([
                    enhanced_mol,  # Global molecular features
                    aggregated_atoms,  # Direct atom aggregation
                    pharm_placeholder  # Placeholder for pharm features
                ], dim=-1)
            else:
                # Edge case: no atoms either (shouldn't happen, but handle it)
                pharm_placeholder = torch.zeros(mol_feat.size(0), self.hidden_channels, device=mol_feat.device)
                atom_placeholder = torch.zeros(mol_feat.size(0), self.hidden_channels, device=mol_feat.device)

                final_feat = torch.cat([
                    mol_feat,
                    atom_placeholder,
                    pharm_placeholder
                ], dim=-1)

        return self.final_proj(final_feat)


# ---------------------------------------
# Feature Attention (Original, preserved)
# ---------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, batch, size=None):
        max_result = scatter(x, batch, dim=0, dim_size=size, reduce='max')
        sum_result = scatter(x, batch, dim=0, dim_size=size, reduce='sum')
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.sigmoid(max_out + sum_out)
        y = y[batch]
        return x * y


# ---------------------------------------
# Descriptor Reconstructor (Preserved from original)
# ---------------------------------------
class DescriptorReconstructor(nn.Module):
    def __init__(self, input_dim, num_descriptors, hidden_dim=64, dropout=0.1):
        super(DescriptorReconstructor, self).__init__()
        self.input_dim = input_dim
        self.num_descriptors = num_descriptors

        if num_descriptors > 0:
            self.reconstructor = Sequential(
                Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_dim // 2, num_descriptors),
                nn.Sigmoid()
            )
        else:
            self.reconstructor = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.reconstructor is not None:
            for module in self.reconstructor:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def forward(self, mol_vec):
        if self.reconstructor is None:
            return None
        return self.reconstructor(mol_vec)


# ---------------------------------------
# NTN Conv (Enhanced with attention extraction)
# ---------------------------------------
class NTNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, slices, dropout, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NTNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slices = slices
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
        if edge_dim is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_dim, out_channels))
        else:
            self.weight_edge = self.register_parameter('weight_edge', None)

        self.bilinear = Bilinear(out_channels, out_channels, slices, bias=False)

        if self.edge_dim is not None:
            self.linear = Linear(3 * out_channels, slices)
        else:
            self.linear = Linear(2 * out_channels, slices)

        # Store attention weights and edge indices for extraction
        self._alpha = None
        self._last_edge_index = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        glorot(self.weight_edge)
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        x = torch.matmul(x, self.weight_node)

        if self.weight_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.weight_edge)

        # Store edge index for attention extraction
        self._last_edge_index = edge_index

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        score = self.bilinear(x_i, x_j)
        if edge_attr is not None:
            vec = torch.cat((x_i, edge_attr, x_j), 1)
            block_score = self.linear(vec)
        else:
            vec = torch.cat((x_i, x_j), 1)
            block_score = self.linear(vec)
        scores = score + block_score
        alpha = torch.tanh(scores)

        # Store attention weights for extraction
        self._alpha = alpha.detach()

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dim_split = self.out_channels // self.slices
        out = torch.max(x_j, edge_attr).view(-1, self.slices, dim_split) if edge_attr is not None else x_j.view(-1,
                                                                                                                self.slices,
                                                                                                                dim_split)
        out = out * alpha.view(-1, self.slices, 1)
        out = out.view(-1, self.out_channels)
        return out


# ---------------------------------------
# Enhanced HiGNN with Three-Level Hierarchy
# ---------------------------------------
def build_model(cfg):
    descriptor_enable = False
    num_descriptors = 0
    descriptor_hidden = 64
    descriptor_dropout = 0.1

    if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE:
        descriptor_enable = True
        num_descriptors = len(cfg.DESCRIPTOR.TYPES)
        descriptor_hidden = cfg.DESCRIPTOR.HIDDEN_DIM
        descriptor_dropout = cfg.DESCRIPTOR.DROPOUT

    # Check for three-level hierarchy configuration
    use_three_level = False
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'THREE_LEVEL'):
        use_three_level = cfg.MODEL.THREE_LEVEL.ENABLE

    # Fixed: Set correct input dimensions
    # Atom features: 28 dimensions (with pharmaco and scaffold features)
    # Pharmacophore features: 8 dimensions
    model = EnhancedHiGNN(
        in_channels=28,  # Atom feature dimension
        pharm_in_channels=8,  # Pharmacophore feature dimension
        hidden_channels=cfg.MODEL.HID,
        out_channels=cfg.MODEL.OUT_DIM,
        edge_dim=10,
        num_layers=cfg.MODEL.DEPTH,
        dropout=cfg.MODEL.DROPOUT,
        slices=cfg.MODEL.SLICES,
        f_att=cfg.MODEL.F_ATT,
        r=cfg.MODEL.R,
        brics=cfg.MODEL.BRICS,
        cl=cfg.LOSS.CL_LOSS,
        descriptor_enable=descriptor_enable,
        num_descriptors=num_descriptors,
        descriptor_hidden=descriptor_hidden,
        descriptor_dropout=descriptor_dropout,
        use_three_level=use_three_level,
        num_heads=cfg.MODEL.THREE_LEVEL.NUM_HEADS if use_three_level else 4
    )

    return model


class EnhancedHiGNN(torch.nn.Module):
    """
    Enhanced HiGNN with optional three-level hierarchy (atom-pharmacophore-molecule)
    """

    def __init__(self, in_channels, pharm_in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 slices, dropout, f_att=False, r=4, brics=True, cl=False,
                 descriptor_enable=False, num_descriptors=0, descriptor_hidden=64,
                 descriptor_dropout=0.1, use_three_level=False, num_heads=4):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.f_att = f_att
        self.brics = brics
        self.cl = cl
        self.descriptor_enable = descriptor_enable
        self.use_three_level = use_three_level

        # Atom feature transformation
        self.lin_a = Linear(in_channels, hidden_channels)
        self.lin_b = Linear(edge_dim, hidden_channels)

        # Pharmacophore feature transformation (separate from atom features)
        if self.use_three_level:
            self.lin_pharm = Linear(pharm_in_channels, hidden_channels)
            self.lin_pharm_edge = Linear(3, hidden_channels)  # Pharm edge features are 3-dimensional

        # Atom-level convolutions
        self.atom_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = NTNConv(hidden_channels, hidden_channels, slices=slices,
                           dropout=dropout, edge_dim=hidden_channels)
            self.atom_convs.append(conv)

        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)

        if self.f_att:
            self.feature_att = FeatureAttention(channels=hidden_channels, reduction=r)

        # Three-level hierarchy components
        if self.use_three_level:
            # Pharmacophore-level processing
            self.pharm_convs = torch.nn.ModuleList()
            for _ in range(num_layers // 2):  # Fewer layers for pharmacophore
                conv = NTNConv(hidden_channels, hidden_channels, slices=slices,
                               dropout=dropout, edge_dim=hidden_channels)
                self.pharm_convs.append(conv)

            # Junction view builder
            self.junction_builder = JunctionViewBuilder(hidden_channels)

            # Three-level fusion
            self.tri_level_fusion = TriLevelHierarchicalFusion(
                hidden_channels, num_heads=num_heads, dropout=dropout
            )

            # Output layers for three-level
            self.out = Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, out_channels)
            )
        else:
            # Original two-level architecture
            if self.brics:
                self.cross_att = GATConv(hidden_channels, hidden_channels, heads=4,
                                         dropout=dropout, add_self_loops=False,
                                         negative_slope=0.01, concat=False)
                self.out = Sequential(
                    Linear(2 * hidden_channels, hidden_channels),
                    Linear(hidden_channels, out_channels)
                )
            else:
                self.out = Linear(hidden_channels, out_channels)

        if self.cl:
            self.lin_project = Linear(hidden_channels, int(hidden_channels / 2))

        # Descriptor reconstructor
        if self.descriptor_enable:
            self.descriptor_reconstructor = DescriptorReconstructor(
                input_dim=hidden_channels,
                num_descriptors=num_descriptors,
                hidden_dim=descriptor_hidden,
                dropout=descriptor_dropout
            )
        else:
            self.descriptor_reconstructor = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_a.reset_parameters()
        self.lin_b.reset_parameters()

        if self.use_three_level:
            self.lin_pharm.reset_parameters()
            self.lin_pharm_edge.reset_parameters()

        for conv in self.atom_convs:
            conv.reset_parameters()

        self.lin_gate.reset_parameters()

        if self.f_att:
            self.feature_att.reset_parameters()

        if self.use_three_level:
            for conv in self.pharm_convs:
                conv.reset_parameters()
            self.junction_builder.reset_parameters()
            self.tri_level_fusion.reset_parameters()
            for module in self.out:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
        else:
            if self.brics:
                self.cross_att.reset_parameters()
                for module in self.out:
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
            else:
                self.out.reset_parameters()

        if self.cl:
            self.lin_project.reset_parameters()

        if self.descriptor_reconstructor is not None:
            self.descriptor_reconstructor.reset_parameters()

    def forward(self, data):
        # Get molecular input
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = F.relu(self.lin_a(x))
        edge_attr = F.relu(self.lin_b(edge_attr))

        # Atom-level convolutions
        for i in range(self.num_layers):
            h = F.relu(self.atom_convs[i](x, edge_index, edge_attr))
            beta = self.lin_gate(torch.cat([x, h, x - h], 1)).sigmoid()
            x = beta * x + (1 - beta) * h
            if self.f_att:
                x = self.feature_att(x, batch)

        mol_vec = global_add_pool(x, batch).relu_()

        if self.use_three_level and hasattr(data, 'pharm_x') and data.pharm_x is not None:
            # Three-level hierarchy processing
            pharm_x = data.pharm_x
            pharm_edge_index = data.pharm_edge_index
            pharm_edge_attr = data.pharm_edge_attr
            pharm_batch = data.pharm_batch
            atom_to_pharm = data.atom_to_pharm

            # CRITICAL FIX: Check if pharmacophores actually exist
            has_pharms = pharm_x is not None and pharm_x.numel() > 0

            if has_pharms:
                # Process pharmacophore features with separate linear layer
                pharm_x = F.relu(self.lin_pharm(pharm_x))
                pharm_edge_attr = F.relu(
                    self.lin_pharm_edge(pharm_edge_attr)) if pharm_edge_attr.numel() > 0 else pharm_edge_attr

                for conv in self.pharm_convs:
                    if pharm_edge_index.numel() > 0:
                        pharm_h = F.relu(
                            conv(pharm_x, pharm_edge_index, pharm_edge_attr if pharm_edge_attr.numel() > 0 else None))
                        beta = self.lin_gate(torch.cat([pharm_x, pharm_h, pharm_x - pharm_h], 1)).sigmoid()
                        pharm_x = beta * pharm_x + (1 - beta) * pharm_h

                # Build junction view
                junction_edges, junction_weights = self.junction_builder(x, pharm_x, atom_to_pharm)

                # Create junction graph structure
                junction_graph = {
                    'edges': junction_edges,
                    'node_types': torch.cat([
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device),  # Atoms
                        torch.ones(pharm_x.size(0), dtype=torch.long, device=x.device),  # Pharmacophores
                        2 * torch.ones(mol_vec.size(0), dtype=torch.long, device=x.device)  # Molecules
                    ]),
                    'edge_types': torch.zeros(junction_edges.size(1), dtype=torch.long, device=x.device)
                }

                # Three-level fusion
                pharm_to_mol_edges = torch.stack(
                    [pharm_batch, torch.arange(mol_vec.size(0), device=mol_vec.device)[pharm_batch]])

                fused_mol_vec = self.tri_level_fusion(
                    x, pharm_x, mol_vec,
                    atom_to_pharm if atom_to_pharm.numel() > 0 else torch.zeros((2, 0), dtype=torch.long,
                                                                                device=x.device),
                    pharm_to_mol_edges, junction_graph
                )
            else:
                # No pharmacophores - use simplified fusion
                # Create empty pharmacophore structure for fusion
                empty_pharm_x = torch.zeros((0, self.hidden_channels), device=x.device)
                empty_junction_graph = {
                    'edges': torch.zeros((2, 0), dtype=torch.long, device=x.device),
                    'node_types': torch.cat([
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                        2 * torch.ones(mol_vec.size(0), dtype=torch.long, device=x.device)
                    ]),
                    'edge_types': torch.zeros(0, dtype=torch.long, device=x.device)
                }

                fused_mol_vec = self.tri_level_fusion(
                    x, empty_pharm_x, mol_vec,
                    torch.zeros((2, 0), dtype=torch.long, device=x.device),
                    torch.zeros((2, 0), dtype=torch.long, device=x.device),
                    empty_junction_graph
                )

            # Output
            out_dropout = F.dropout(fused_mol_vec, p=self.dropout, training=self.training)
            main_pred = self.out(out_dropout)

            # Descriptor reconstruction
            descriptor_pred = None
            if self.descriptor_enable and self.descriptor_reconstructor is not None:
                descriptor_pred = self.descriptor_reconstructor(fused_mol_vec)

            # Return results
            if self.cl:
                mol_vec_proj = self.lin_project(fused_mol_vec).relu_()
                # CRITICAL FIX: Handle empty pharmacophore batch for projection
                if has_pharms and pharm_batch.numel() > 0:
                    pharm_vec_proj = self.lin_project(global_add_pool(pharm_x, pharm_batch)).relu_()
                else:
                    # Create zero vector for empty pharmacophores
                    pharm_vec_proj = torch.zeros((mol_vec_proj.size(0), mol_vec_proj.size(1)),
                                                 device=mol_vec_proj.device)

                if self.descriptor_enable:
                    return {
                        'main': main_pred,
                        'mol_vec': mol_vec_proj,
                        'fra_vec': pharm_vec_proj,
                        'descriptors': descriptor_pred
                    }
                else:
                    return main_pred, mol_vec_proj, pharm_vec_proj
            else:
                if self.descriptor_enable:
                    return {
                        'main': main_pred,
                        'descriptors': descriptor_pred
                    }
                else:
                    return main_pred

        elif self.brics:
            # Original two-level processing (fallback)
            fra_x = data.x
            fra_edge_index = data.fra_edge_index
            fra_edge_attr = data.fra_edge_attr
            cluster = data.cluster_index

            fra_x = F.relu(self.lin_a(fra_x))
            fra_edge_attr = F.relu(self.lin_b(fra_edge_attr))

            for i in range(self.num_layers):
                h = F.relu(self.atom_convs[i](fra_x, fra_edge_index, fra_edge_attr))
                beta = self.lin_gate(torch.cat([fra_x, h, fra_x - h], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * h
                if self.f_att:
                    fra_x = self.feature_att(fra_x, cluster)

            fra_x = global_add_pool(fra_x, cluster).relu_()

            cluster, perm = consecutive_cluster(cluster)
            fra_batch = pool_batch(perm, data.batch)

            row = torch.arange(fra_batch.size(0), device=batch.device)
            mol_fra_index = torch.stack([row, fra_batch], dim=0)
            fra_vec = self.cross_att((fra_x, mol_vec), mol_fra_index).relu_()

            vectors_concat = [mol_vec, fra_vec]
            out = torch.cat(vectors_concat, 1)

            out_dropout = F.dropout(out, p=self.dropout, training=self.training)
            main_pred = self.out(out_dropout)

            descriptor_pred = None
            if self.descriptor_enable and self.descriptor_reconstructor is not None:
                descriptor_pred = self.descriptor_reconstructor(mol_vec)

            if self.cl:
                mol_vec_proj = self.lin_project(mol_vec).relu_()
                fra_vec_proj = self.lin_project(fra_vec).relu_()

                if self.descriptor_enable:
                    return {
                        'main': main_pred,
                        'mol_vec': mol_vec_proj,
                        'fra_vec': fra_vec_proj,
                        'descriptors': descriptor_pred
                    }
                else:
                    return main_pred, mol_vec_proj, fra_vec_proj
            else:
                if self.descriptor_enable:
                    return {
                        'main': main_pred,
                        'descriptors': descriptor_pred
                    }
                else:
                    return main_pred

        else:
            # No BRICS fragments
            assert self.cl is False
            out_dropout = F.dropout(mol_vec, p=self.dropout, training=self.training)
            main_pred = self.out(out_dropout)

            descriptor_pred = None
            if self.descriptor_enable and self.descriptor_reconstructor is not None:
                descriptor_pred = self.descriptor_reconstructor(mol_vec)

            if self.descriptor_enable:
                return {
                    'main': main_pred,
                    'descriptors': descriptor_pred
                }
            else:
                return main_pred


# ---------------------------------------
# Attention Extraction Utility Functions
# ---------------------------------------
def extract_attention_weights_from_model(model):
    """
    Extract all attention weights from a model after forward pass
    """
    attention_data = {}

    # Extract from atom convolutions (NTN layers)
    atom_attentions = []
    for i, conv in enumerate(model.atom_convs):
        if hasattr(conv, '_alpha') and conv._alpha is not None:
            attention_data[f'atom_conv_{i}'] = {
                'attention': conv._alpha.cpu().numpy(),
                'edge_index': conv._last_edge_index.cpu().numpy() if conv._last_edge_index is not None else None
            }
            atom_attentions.append(conv._alpha.cpu().numpy())

    if atom_attentions:
        attention_data['atom_combined'] = np.mean(atom_attentions, axis=0)

    # Extract from pharmacophore convolutions if present
    if hasattr(model, 'pharm_convs'):
        pharm_attentions = []
        for i, conv in enumerate(model.pharm_convs):
            if hasattr(conv, '_alpha') and conv._alpha is not None:
                attention_data[f'pharm_conv_{i}'] = {
                    'attention': conv._alpha.cpu().numpy(),
                    'edge_index': conv._last_edge_index.cpu().numpy() if conv._last_edge_index is not None else None
                }
                pharm_attentions.append(conv._alpha.cpu().numpy())

        if pharm_attentions:
            attention_data['pharm_combined'] = np.mean(pharm_attentions, axis=0)

    # Extract from multi-view attention if present
    if hasattr(model, 'tri_level_fusion'):
        fusion = model.tri_level_fusion

        if hasattr(fusion, 'atom_to_pharm_attn') and hasattr(fusion.atom_to_pharm_attn, '_last_attention_weights'):
            if fusion.atom_to_pharm_attn._last_attention_weights is not None:
                attention_data['atom_to_pharm'] = fusion.atom_to_pharm_attn._last_attention_weights.cpu().numpy()

        if hasattr(fusion, 'cross_scale_fusion') and hasattr(fusion.cross_scale_fusion, '_last_attention_weights'):
            if fusion.cross_scale_fusion._last_attention_weights is not None:
                attention_data['cross_scale'] = fusion.cross_scale_fusion._last_attention_weights.cpu().numpy()

    return attention_data


def register_attention_hooks(model):
    """
    Register hooks to capture attention weights during forward pass
    """
    hooks = []
    attention_storage = {}

    def create_hook(name):
        def hook(module, input, output):
            if hasattr(module, '_alpha') and module._alpha is not None:
                attention_storage[name] = module._alpha.detach().cpu()
            elif hasattr(module, '_last_attention_weights') and module._last_attention_weights is not None:
                attention_storage[name] = module._last_attention_weights.detach().cpu()

        return hook

    # Register hooks on atom convolutions
    for i, conv in enumerate(model.atom_convs):
        hook = conv.register_forward_hook(create_hook(f'atom_conv_{i}'))
        hooks.append(hook)

    # Register hooks on pharmacophore convolutions if present
    if hasattr(model, 'pharm_convs'):
        for i, conv in enumerate(model.pharm_convs):
            hook = conv.register_forward_hook(create_hook(f'pharm_conv_{i}'))
            hooks.append(hook)

    # Register hooks on three-level fusion if present
    if hasattr(model, 'tri_level_fusion'):
        fusion = model.tri_level_fusion

        if hasattr(fusion, 'atom_to_pharm_attn'):
            hook = fusion.atom_to_pharm_attn.register_forward_hook(create_hook('atom_to_pharm'))
            hooks.append(hook)

        if hasattr(fusion, 'cross_scale_fusion'):
            hook = fusion.cross_scale_fusion.register_forward_hook(create_hook('cross_scale'))
            hooks.append(hook)

    return hooks, attention_storage


def remove_attention_hooks(hooks):
    """
    Remove registered hooks
    """
    for hook in hooks:
        hook.remove()


def extract_model_attention(model, data):
    """
    Complete example of extracting attention from model
    """
    model.eval()

    # Register hooks
    hooks, attention_storage = register_attention_hooks(model)

    try:
        # Forward pass
        with torch.no_grad():
            output = model(data)

        # Extract attention weights
        attention_data = extract_attention_weights_from_model(model)

        # Add hook-captured attention
        attention_data.update(attention_storage)

        return attention_data, output

    finally:
        # Clean up hooks
        remove_attention_hooks(hooks)