                       


import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import Random
from collections import defaultdict
from datetime import datetime

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from utils import get_task_names

RDLogger.DisableLog('rdApp.*')

DEFAULT_SIMILARITY_THRESHOLD = 0.7


                                            
                            
                                            

def get_cache_key(dataset, seed, split_type, descriptor_types, use_pharmacophores,
                  rate=None, mw_bins=None, similarity_threshold=None):

    desc_key = "_".join(sorted(descriptor_types)) if descriptor_types else "none"
    pharm_key = "_pharm" if use_pharmacophores else ""

    if split_type == 'noise' and rate is not None:
        return f"{dataset}_seed_{seed}_{split_type}_{int(100*rate)}_{desc_key}{pharm_key}"
    elif split_type == 'scaffold':
        return f"{dataset}_seed_{seed}_scaffold_{desc_key}{pharm_key}"
    elif split_type == 'scaffoldrandom':
        return f"{dataset}_seed_{seed}_scaffoldrandom_{desc_key}{pharm_key}"
    elif split_type == 'molecular_weight':
        mw_str = "_".join(str(b) for b in mw_bins) if mw_bins else "default"
        return f"{dataset}_seed_{seed}_mw_{mw_str}_{desc_key}{pharm_key}"
    elif split_type == 'similarity':
        sim_thr = similarity_threshold if similarity_threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
        sim_str = str(int(sim_thr * 100))
        return f"{dataset}_seed_{seed}_similarity_{sim_str}_{desc_key}{pharm_key}"
    else:          
        return f"{dataset}_seed_{seed}_{desc_key}{pharm_key}"


def get_cache_metadata(cache_key, dataset_config):

    return {
        'cache_key': cache_key,
        'creation_time': datetime.now().isoformat(),
        'config': dataset_config,
        'version': '2.0'                                          
    }


def save_cache_with_metadata(save_path, train_data, val_data, test_data, metadata):

    cache_data = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': metadata
    }
    torch.save(cache_data, save_path)


def load_cache_with_validation(save_path, logger=None):

    try:
        cache_data = torch.load(save_path,weights_only=False)
        
                                                    
        if isinstance(cache_data, list):
            if logger:
                logger.warning(f"Loading old cache format from {save_path}")
            return cache_data[0], cache_data[1], cache_data[2], None
        
                                        
        train_data = cache_data['train']
        val_data = cache_data['val']
        test_data = cache_data['test']
        metadata = cache_data.get('metadata', None)
        
        if metadata and logger:
            logger.info(f"Cache metadata:")
            logger.info(f"  Created: {metadata.get('creation_time', 'unknown')}")
            logger.info(f"  Version: {metadata.get('version', 'unknown')}")
        
        return train_data, val_data, test_data, metadata
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading cache: {e}")
        raise


def check_processed_data_exists(root, dataset, use_pharmacophores):

    suffix = '_pharm' if use_pharmacophores else ''
    processed_file = os.path.join(root, 'processed', f'{dataset}{suffix}.pt')
    return os.path.isfile(processed_file)


                                            
                                
                                            

def _get_data_attr(dataset, attr_name):

    if hasattr(dataset, '_data') and dataset._data is not None:
        return getattr(dataset._data, attr_name)
    elif hasattr(dataset, 'data') and dataset.data is not None:
        return getattr(dataset.data, attr_name)
    elif hasattr(dataset, f'_{attr_name}'):
        return getattr(dataset, f'_{attr_name}')
    else:
        raise AttributeError(f"Cannot access {attr_name} from dataset")


def _set_data_attr(dataset, attr_name, value):

    if hasattr(dataset, '_data') and dataset._data is not None:
        setattr(dataset._data, attr_name, value)
    elif hasattr(dataset, 'data') and dataset.data is not None:
        setattr(dataset.data, attr_name, value)
    else:
        setattr(dataset, f'_{attr_name}', value)


def _del_data_attr(dataset, attr_name):

    try:
        if hasattr(dataset, '_data') and dataset._data is not None and hasattr(dataset._data, attr_name):
            delattr(dataset._data, attr_name)
        if hasattr(dataset, 'data') and dataset.data is not None and hasattr(dataset.data, attr_name):
            delattr(dataset.data, attr_name)
    except:
        pass


                                            
                                  
                                            

def extract_pharmacophores(mol):

    pharm_patterns = {
        'Hbond_donor': Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]'),
        'Hbond_acceptor': Chem.MolFromSmarts(
            '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),'
            '$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]'
        ),
        'Aromatic': Chem.MolFromSmarts('a'),
        'Hydrophobic': Chem.MolFromSmarts('[C,c,F,Cl,Br,I,#1]'),
        'Positive': Chem.MolFromSmarts('[+,$([N;H2&+0]),$(N;H3&+0)]'),
        'Negative': Chem.MolFromSmarts('[-,$([O,S;H1;-0])]'),
        'Metal_binding': Chem.MolFromSmarts('[O,S,N]')
    }
    
    pharmacophores = []
    atom_to_pharm = defaultdict(list)
    
    for pharm_type, pattern in pharm_patterns.items():
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        
        for match_idx, match in enumerate(matches):
            pharm_id = len(pharmacophores)
            
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                positions = [conf.GetAtomPosition(atom_idx) for atom_idx in match]
                center_x = sum(pos.x for pos in positions) / len(positions)
                center_y = sum(pos.y for pos in positions) / len(positions)
                center_z = sum(pos.z for pos in positions) / len(positions)
                center = [center_x, center_y, center_z]
            else:
                center = [0.0, 0.0, 0.0]
            
            pharm_features = {
                'type': pharm_type,
                'type_encoding': encode_pharm_type(pharm_type),
                'center': center,
                'size': len(match),
                'atoms': list(match)
            }
            
            pharmacophores.append(pharm_features)
            
            for atom_idx in match:
                atom_to_pharm[atom_idx].append(pharm_id)
    
    return pharmacophores, dict(atom_to_pharm)


def encode_pharm_type(pharm_type):

    types = ['Hbond_donor', 'Hbond_acceptor', 'Aromatic', 'Hydrophobic', 
             'Positive', 'Negative', 'Metal_binding']
    encoding = [0] * len(types)
    if pharm_type in types:
        encoding[types.index(pharm_type)] = 1
    return encoding


def build_pharmacophore_graph(pharmacophores, distance_threshold=5.0):

    edges = []
    edge_attrs = []
    
    for i in range(len(pharmacophores)):
        for j in range(i + 1, len(pharmacophores)):
            center_i = pharmacophores[i]['center']
            center_j = pharmacophores[j]['center']
            
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(center_i, center_j)))
            
            if all(c == 0.0 for c in center_i) and all(c == 0.0 for c in center_j):
                shared_atoms = set(pharmacophores[i]['atoms']) & set(pharmacophores[j]['atoms'])
                compatible = (pharmacophores[i]['type'], pharmacophores[j]['type']) in [
                    ('Hbond_donor', 'Hbond_acceptor'),
                    ('Positive', 'Negative'),
                    ('Aromatic', 'Hydrophobic')
                ]
                if shared_atoms or compatible:
                    dist = 1.0
                else:
                    continue
            
            if dist < distance_threshold:
                edges.append([i, j])
                edges.append([j, i])
                
                edge_attr = [
                    1.0 / (1.0 + dist),
                    float(pharmacophores[i]['type'] == pharmacophores[j]['type']),
                    float(set(pharmacophores[i]['atoms']) & set(pharmacophores[j]['atoms']) != set())
                ]
                edge_attrs.append(edge_attr)
                edge_attrs.append(edge_attr)
    
    return edges, edge_attrs


                                            
                       
                                            

def calculate_molecular_descriptors(mol, descriptor_types):

    descriptors = []
    
    for desc_type in descriptor_types:
        if desc_type == 'mw':
            descriptors.append(Descriptors.MolWt(mol))
        elif desc_type == 'logp':
            descriptors.append(Descriptors.MolLogP(mol))
        elif desc_type == 'tpsa':
            descriptors.append(Descriptors.TPSA(mol))
        elif desc_type == 'hbd':
            descriptors.append(Descriptors.NumHDonors(mol))
        elif desc_type == 'hba':
            descriptors.append(Descriptors.NumHAcceptors(mol))
        elif desc_type == 'nrb':
            descriptors.append(Descriptors.NumRotatableBonds(mol))
        elif desc_type == 'aromatic_rings':
            descriptors.append(Descriptors.NumAromaticRings(mol))
        elif desc_type == 'heteroatoms':
            descriptors.append(Descriptors.NumHeteroatoms(mol))
        elif desc_type == 'flexibility':
            n_bonds = mol.GetNumBonds()
            n_rot_bonds = Descriptors.NumRotatableBonds(mol)
            flexibility = n_rot_bonds / max(n_bonds, 1)
            descriptors.append(flexibility)
        elif desc_type == 'complexity':
            descriptors.append(Descriptors.BertzCT(mol))
        else:
            descriptors.append(0.0)
    
    return np.array(descriptors, dtype=np.float32)


def normalize_descriptors(descriptors, descriptor_types):

    normalized = descriptors.copy()
    
    normalization_ranges = {
        'mw': [0, 2000],
        'logp': [-5, 10],
        'tpsa': [0, 400],
        'hbd': [0, 15],
        'hba': [0, 20],
        'nrb': [0, 30],
        'aromatic_rings': [0, 8],
        'heteroatoms': [0, 20],
        'flexibility': [0, 1],
        'complexity': [0, 2000]
    }
    
    for i, desc_type in enumerate(descriptor_types):
        if desc_type in normalization_ranges:
            min_val, max_val = normalization_ranges[desc_type]
            normalized[i] = np.clip((descriptors[i] - min_val) / (max_val - min_val), 0, 1)
    
    return normalized


                                            
                        
                                            

def onehot_encoding(x, allowable_set):

    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


fun_smarts = {
    'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
    'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
    'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
    'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
    'Halogen': '[F,Cl,Br,I]'
}
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(mol):

    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


def tag_scaffold(mol):

    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol


def atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):

    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)
    
    feat = []
    for atom in mol.GetAtoms():
        element_feat = onehot_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'other']
        )
        
        degree_feat = onehot_encoding_unk(
            atom.GetDegree(),
            [1, 2, 3, 4, 'other']
        )
        
        charge_feat = [atom.GetFormalCharge()]
        aromatic_feat = [atom.GetIsAromatic()]
        
        hybrid_feat = onehot_encoding_unk(
            atom.GetHybridization(),
            [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
             Chem.rdchem.HybridizationType.SP3, 'other']
        )
        
        ring_feat = [atom.IsInRing()]
        
        pharmaco_feat = []
        if pharmaco:
            pharmaco_feat = [
                int(atom.GetProp('Hbond_donor')),
                int(atom.GetProp('Hbond_acceptor')),
                int(atom.GetProp('Halogen'))
            ]
        
        scaffold_feat = [int(atom.GetProp('Scaffold'))] if scaffold else []
        
        chirality_feat = []
        if use_chirality:
            try:
                chirality_feat = onehot_encoding_unk(
                    atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                chirality_feat = [0, 0] + [atom.HasProp('_ChiralityPossible')]
        
        total_feat = (element_feat + degree_feat + charge_feat + aromatic_feat +
                     hybrid_feat + ring_feat + pharmaco_feat + scaffold_feat + chirality_feat)
        
        feat.append(total_feat)
    
    return np.array(feat)


def bond_attr(mol, use_chirality=True):

    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


def bond_break(mol):

    results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.int64)

    if results.size == 0:
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(mol)
    else:
        bond_to_break = results[:, 0, :]
        bond_to_break = bond_to_break.tolist()
        with Chem.RWMol(mol) as rwmol:
            for i in bond_to_break:
                rwmol.RemoveBond(*i)
        rwmol = rwmol.GetMol()
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(rwmol)
        cluster_idx = torch.LongTensor(cluster_idx)

    return fra_edge_index, fra_edge_attr, cluster_idx


                                            
                     
                                            

class EnhancedMolData(Data):

    def __init__(self, 
                 fra_edge_index=None, fra_edge_attr=None, cluster_index=None,
                 descriptors=None,
                 pharm_x=None, pharm_edge_index=None, pharm_edge_attr=None,
                 pharm_batch=None, atom_to_pharm=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.cluster_index = cluster_index
        self.fra_edge_index = fra_edge_index
        self.fra_edge_attr = fra_edge_attr
        self.descriptors = descriptors
        
        self.pharm_x = pharm_x
        self.pharm_edge_index = pharm_edge_index
        self.pharm_edge_attr = pharm_edge_attr
        self.pharm_batch = pharm_batch
        self.atom_to_pharm = atom_to_pharm

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster_index':
            if self.cluster_index is not None and self.cluster_index.numel() > 0:
                return int(self.cluster_index.max()) + 1
            else:
                return 0
        elif key == 'pharm_batch':
            if self.pharm_batch is not None and self.pharm_batch.numel() > 0:
                return int(self.pharm_batch.max()) + 1
            else:
                return 0
        elif key == 'atom_to_pharm':
            if self.atom_to_pharm is not None and self.atom_to_pharm.numel() > 0:
                num_atoms = self.x.size(0) if self.x is not None else 0
                num_pharms = self.pharm_x.size(0) if self.pharm_x is not None else 0
                return torch.tensor([num_atoms, num_pharms])
            else:
                return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


class EnhancedMolDataset(InMemoryDataset):

    def __init__(self, root, dataset, task_type, tasks, 
                 descriptor_types=None, use_pharmacophores=False,
                 logger=None, transform=None, pre_transform=None, pre_filter=None):
        
        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.descriptor_types = descriptor_types or []
        self.use_pharmacophores = use_pharmacophores
        self.logger = logger

                                                
        processed_exists = check_processed_data_exists(root, dataset, use_pharmacophores)
        
        if processed_exists and logger:
            logger.info(f"✓ Found existing processed data for {dataset}")
        elif logger:
            logger.info(f"× No processed data found, will create from raw data")

        super().__init__(root, transform, pre_transform, pre_filter)
        
                                                            
        try:
            start_time = time.time()
            self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
            load_time = time.time() - start_time
            
            if self.logger:
                self.logger.info(f"✓ Loaded processed data in {load_time:.2f}s")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading dataset: {e}")
            raise

    @property
    def raw_file_names(self):
        return [f'{self.dataset}.csv']

    @property
    def processed_file_names(self):
        suffix = '_pharm' if self.use_pharmacophores else ''
        return [f'{self.dataset}{suffix}.pt']

    def download(self):
        pass

    def process(self):

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("PROCESSING RAW DATA")
            self.logger.info("=" * 60)
        
        start_time = time.time()
        
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        if self.logger:
            self.logger.info(f'Total SMILES to process: {len(smilesList)}')
        
        remained_smiles = []
        canonical_smiles_list = []
        
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                if self.logger:
                    self.logger.warning(f'Failed to process: {smiles}')

        if self.logger:
            self.logger.info(f'Successfully parsed: {len(remained_smiles)}/{len(smilesList)} SMILES')

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList, desc="Processing molecules")):
            mol = Chem.MolFromSmiles(smi)
            
                                                             
            if self.use_pharmacophores:
                try:
                    from rdkit.Chem import AllChem
                    result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50)
                    if result == 0:
                        try:
                            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                        except:
                            pass
                    else:
                        result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=50)
                        if result == 0:
                            try:
                                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                            except:
                                pass
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f'Could not generate 3D for {smi}: {e}')
            
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label = np.array(label, dtype=np.float64)
                label[np.isnan(label)] = 666
                data.y = torch.tensor([label], dtype=torch.float)
                if self.task_type == 'regression':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        process_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"✓ Processing completed in {process_time:.2f}s")
            self.logger.info(f"✓ Saved to {self.processed_paths[0]}")
            self.logger.info("=" * 60)

    def mol2graph(self, mol):

        smiles = Chem.MolToSmiles(mol)
        if mol is None:
            return None
        
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = bond_break(mol)
        
                                            
        descriptors = None
        if self.descriptor_types:
            descriptors_raw = calculate_molecular_descriptors(mol, self.descriptor_types)
            descriptors = normalize_descriptors(descriptors_raw, self.descriptor_types)
            descriptors = torch.FloatTensor(descriptors)
        
                                             
        pharm_data = {}
        if self.use_pharmacophores:
            pharmacophores, atom_to_pharm_dict = extract_pharmacophores(mol)
            
            if len(pharmacophores) > 0:
                pharm_features = []
                for pharm in pharmacophores:
                    feat = pharm['type_encoding'] + [pharm['size'] / 10.0]
                    pharm_features.append(feat)
                
                pharm_x = torch.FloatTensor(pharm_features)
                
                pharm_edges, pharm_edge_attrs = build_pharmacophore_graph(pharmacophores)
                
                if len(pharm_edges) > 0:
                    pharm_edge_index = torch.LongTensor(pharm_edges).t()
                    pharm_edge_attr = torch.FloatTensor(pharm_edge_attrs)
                else:
                    pharm_edge_index = torch.zeros((2, 0), dtype=torch.int64)
                    pharm_edge_attr = torch.zeros((0, 3))
                
                atom_to_pharm_edges = []
                for atom_idx, pharm_ids in atom_to_pharm_dict.items():
                    for pharm_id in pharm_ids:
                        atom_to_pharm_edges.append([atom_idx, pharm_id])
                
                if len(atom_to_pharm_edges) > 0:
                    atom_to_pharm = torch.LongTensor(atom_to_pharm_edges)
                else:
                    atom_to_pharm = torch.zeros((0, 2), dtype=torch.int64)
                
                pharm_batch = torch.zeros(len(pharmacophores), dtype=torch.int64)
                
                pharm_data = {
                    'pharm_x': pharm_x,
                    'pharm_edge_index': pharm_edge_index,
                    'pharm_edge_attr': pharm_edge_attr,
                    'pharm_batch': pharm_batch,
                    'atom_to_pharm': atom_to_pharm
                }
        
        if not isinstance(cluster_index, torch.Tensor):
            cluster_index = torch.LongTensor([]) if cluster_index == [] or cluster_index is None else torch.LongTensor(cluster_index)
        
        data = EnhancedMolData(
            x=torch.FloatTensor(node_attr),
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            fra_edge_index=torch.LongTensor(fra_edge_index).t(),
            fra_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=cluster_index,
            descriptors=descriptors,
            y=None,
            smiles=smiles,
            **pharm_data
        )
        
        return data


                                            
                                                 
                                            

def load_dataset_random(path, dataset, seed, task_type, tasks=None, 
                       descriptor_types=None, use_pharmacophores=False, logger=None):

    
                                 
    cache_key = get_cache_key(dataset, seed, 'random', descriptor_types, use_pharmacophores)
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')
    
                           
    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Random Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")
        
        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time
        
        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)
        
        return trn, val, test
    
                                          
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Random Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")
    
    start_time = time.time()
    
                    
    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )
    
                                                   
    _del_data_attr(pyg_dataset, 'smiles')

                   
    train_size = int(0.8 * len(pyg_dataset))
    val_size = int(0.1 * len(pyg_dataset))
    test_size = len(pyg_dataset) - train_size - val_size

    pyg_dataset = pyg_dataset.shuffle()
    trn = pyg_dataset[:train_size]
    val = pyg_dataset[train_size:(train_size + val_size)]
    test = pyg_dataset[(train_size + val_size):]

    if logger:
        logger.info(f'Dataset split:')
        logger.info(f'  Total: {len(pyg_dataset)} molecules')
        logger.info(f'  Train: {train_size} ({train_size/len(pyg_dataset)*100:.1f}%)')
        logger.info(f'  Val: {val_size} ({val_size/len(pyg_dataset)*100:.1f}%)')
        logger.info(f'  Test: {test_size} ({test_size/len(pyg_dataset)*100:.1f}%)')

                                                
    if task_type == 'classification':
        weights = []
        y_data = _get_data_attr(pyg_dataset, "y")
        for i in range(len(tasks)):
            pos_len = (y_data[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights
        
        if logger:
            logger.info(f'Class weights: {weights}')
    else:
        trn.weights = None

                              
    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'random',
        'task_type': task_type,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }
    
    metadata = get_cache_metadata(cache_key, dataset_config)
    
                                       
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_cache_with_metadata(save_path, trn, val, test, metadata)
    
    process_time = time.time() - start_time
    
    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)
    
                                             
    return load_dataset_random(path, dataset, seed, task_type, tasks, descriptor_types, use_pharmacophores, logger)


                                            
                          
                                            

def generate_scaffold(mol, include_chirality=False):

    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def scaffold_to_smiles(smiles, use_indices=False):

    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)
    return scaffolds


def scaffold_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1), balanced=True, seed=1, logger=None):

    assert sum(sizes) == 1

    if logger:
        logger.info('Generating scaffolds for split...')
    
    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    smiles_data = _get_data_attr(pyg_dataset, "smiles")
    scaffold_to_indices = scaffold_to_smiles(smiles_data, use_indices=True)

    random = Random(seed)

    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:
        index_sets = sorted(list(scaffold_to_indices.values()),
                           key=lambda index_set: len(index_set),
                           reverse=True)

    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size:
            val_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    if logger:
        logger.info(f'Scaffold split results:')
        logger.info(f'  Total scaffolds: {len(scaffold_to_indices)}')
        logger.info(f'  Train scaffolds: {train_scaffold_count} ({len(train_ids)} molecules)')
        logger.info(f'  Val scaffolds: {val_scaffold_count} ({len(val_ids)} molecules)')
        logger.info(f'  Test scaffolds: {test_scaffold_count} ({len(test_ids)} molecules)')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    if task_type == 'classification':
        weights = []
        y_data = _get_data_attr(pyg_dataset, "y")
        for i in range(len(tasks)):
            pos_len = (y_data[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    else:
        weights = None

    return train_ids, val_ids, test_ids, weights


def load_dataset_scaffold(path, dataset, seed, task_type, tasks=None,
                         descriptor_types=None, use_pharmacophores=False, logger=None):

    
                                 
    cache_key = get_cache_key(dataset, seed, 'scaffold', descriptor_types, use_pharmacophores)
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')
    
                           
    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Scaffold Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")
        
        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time
        
        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)
        
        return trn, val, test
    
                                          
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Scaffold Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")
    
    start_time = time.time()
    
                    
    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )

                            
    trn_id, val_id, test_id, weights = scaffold_split(
        pyg_dataset, task_type=task_type, tasks=tasks, seed=seed, logger=logger
    )
    
                             
    _del_data_attr(pyg_dataset, 'smiles')
    
                   
    trn = pyg_dataset[torch.LongTensor(trn_id)]
    val = pyg_dataset[torch.LongTensor(val_id)]
    test = pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

                              
    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'scaffold',
        'task_type': task_type,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': len(trn_id),
        'val_size': len(val_id),
        'test_size': len(test_id)
    }
    
    metadata = get_cache_metadata(cache_key, dataset_config)
    
                                       
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_cache_with_metadata(save_path, trn, val, test, metadata)
    
    process_time = time.time() - start_time
    
    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)
    
                       
    return load_dataset_scaffold(path, dataset, seed, task_type, tasks, descriptor_types, use_pharmacophores, logger)


                                            
                                 
                                            

def scaffoldrandom_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1), seed=1, logger=None):


    assert sum(sizes) == 1

    if logger:
        logger.info('Generating scaffolds for scaffold-random split...')
    
    num = len(pyg_dataset)
    train_size_target = int(sizes[0] * num)
    val_size_target = int(sizes[1] * num)

    smiles_data = _get_data_attr(pyg_dataset, "smiles")
    scaffold_to_indices = scaffold_to_smiles(smiles_data, use_indices=True)

                                      
    random = Random(seed)
    scaffold_groups = list(scaffold_to_indices.values())
    random.shuffle(scaffold_groups)

    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    for index_set in scaffold_groups:
        if len(train_ids) + len(index_set) <= train_size_target:
            train_ids += list(index_set)
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size_target:
            val_ids += list(index_set)
            val_scaffold_count += 1
        else:
            test_ids += list(index_set)
            test_scaffold_count += 1

                                              
    random.shuffle(train_ids)
    random.shuffle(val_ids)
    random.shuffle(test_ids)

    if logger:
        logger.info(f'Scaffold-random split results:')
        logger.info(f'  Total scaffolds: {len(scaffold_to_indices)}')
        logger.info(f'  Train scaffolds: {train_scaffold_count} ({len(train_ids)} molecules)')
        logger.info(f'  Val scaffolds: {val_scaffold_count} ({len(val_ids)} molecules)')
        logger.info(f'  Test scaffolds: {test_scaffold_count} ({len(test_ids)} molecules)')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    if task_type == 'classification':
        weights = []
        y_data = _get_data_attr(pyg_dataset, "y")
        for i in range(len(tasks)):
            pos_len = (y_data[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    else:
        weights = None

    return train_ids, val_ids, test_ids, weights


def load_dataset_scaffoldrandom(path, dataset, seed, task_type, tasks=None,
                                descriptor_types=None, use_pharmacophores=False, logger=None):

    
                                 
    cache_key = get_cache_key(dataset, seed, 'scaffoldrandom', descriptor_types, use_pharmacophores)
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')
    
                           
    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Scaffold-Random Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")
        
        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time
        
        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)
        
        return trn, val, test
    
                                          
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Scaffold-Random Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")
    
    start_time = time.time()
    
                    
    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )

                                   
    trn_id, val_id, test_id, weights = scaffoldrandom_split(
        pyg_dataset, task_type=task_type, tasks=tasks, seed=seed, logger=logger
    )
    
                             
    _del_data_attr(pyg_dataset, 'smiles')
    
                   
    trn = pyg_dataset[torch.LongTensor(trn_id)]
    val = pyg_dataset[torch.LongTensor(val_id)]
    test = pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

                              
    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'scaffoldrandom',
        'task_type': task_type,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': len(trn_id),
        'val_size': len(val_id),
        'test_size': len(test_id)
    }
    
    metadata = get_cache_metadata(cache_key, dataset_config)
    
                                       
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_cache_with_metadata(save_path, trn, val, test, metadata)
    
    process_time = time.time() - start_time
    
    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)
    
                       
    return load_dataset_scaffoldrandom(path, dataset, seed, task_type, tasks, descriptor_types, use_pharmacophores, logger)


                                            
                                  
                                            

def molecular_weight_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1),
                           mw_bins=None, seed=1, logger=None):


    assert sum(sizes) == 1

    if mw_bins is None:
        mw_bins = [200, 300, 400, 500]

    if logger:
        logger.info('Computing molecular weights for MW-based split...')
    
    num = len(pyg_dataset)
    smiles_data = _get_data_attr(pyg_dataset, "smiles")
    
                                 
    mol_weights = []
    for smi in smiles_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol_weights.append(Descriptors.MolWt(mol))
        else:
            mol_weights.append(0.0)
    
    mol_weights = np.array(mol_weights)
    
                                                          
    bin_edges = [0] + list(mw_bins) + [float('inf')]
    bin_indices = defaultdict(list)
    
    for idx, mw in enumerate(mol_weights):
        for b in range(len(bin_edges) - 1):
            if bin_edges[b] <= mw < bin_edges[b + 1]:
                bin_indices[b].append(idx)
                break
    
    if logger:
        logger.info(f'Molecular weight distribution:')
        for b in range(len(bin_edges) - 1):
            low = bin_edges[b]
            high = bin_edges[b + 1]
            count = len(bin_indices[b])
            high_str = f'{high:.0f}' if high != float('inf') else '∞'
            logger.info(f'  MW [{low:.0f}, {high_str}): {count} molecules')
    
                                          
    random_gen = Random(seed)
    train_ids, val_ids, test_ids = [], [], []
    
    for b in sorted(bin_indices.keys()):
        indices = bin_indices[b]
        random_gen.shuffle(indices)
        
        n = len(indices)
        n_train = int(sizes[0] * n)
        n_val = int(sizes[1] * n)
        
        train_ids.extend(indices[:n_train])
        val_ids.extend(indices[n_train:n_train + n_val])
        test_ids.extend(indices[n_train + n_val:])
    
                                     
    random_gen.shuffle(train_ids)
    random_gen.shuffle(val_ids)
    random_gen.shuffle(test_ids)

    if logger:
        logger.info(f'Molecular weight split results:')
        logger.info(f'  Train: {len(train_ids)} molecules')
        logger.info(f'  Val: {len(val_ids)} molecules')
        logger.info(f'  Test: {len(test_ids)} molecules')
        
                                     
        for split_name, split_ids in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
            if len(split_ids) > 0:
                split_mws = mol_weights[split_ids]
                logger.info(f'  {split_name} MW: mean={np.mean(split_mws):.1f}, '
                           f'std={np.std(split_mws):.1f}, '
                           f'range=[{np.min(split_mws):.1f}, {np.max(split_mws):.1f}]')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    if task_type == 'classification':
        weights = []
        y_data = _get_data_attr(pyg_dataset, "y")
        for i in range(len(tasks)):
            pos_len = (y_data[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    else:
        weights = None

    return train_ids, val_ids, test_ids, weights


def load_dataset_molecular_weight(path, dataset, seed, task_type, tasks=None,
                                  mw_bins=None, descriptor_types=None,
                                  use_pharmacophores=False, logger=None):

    
                                 
    cache_key = get_cache_key(dataset, seed, 'molecular_weight', descriptor_types, use_pharmacophores, mw_bins=mw_bins)
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')
    
                           
    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Molecular Weight Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")
        
        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time
        
        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)
        
        return trn, val, test
    
                                          
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Molecular Weight Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")
    
    start_time = time.time()
    
                    
    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )

                                    
    trn_id, val_id, test_id, weights = molecular_weight_split(
        pyg_dataset, task_type=task_type, tasks=tasks,
        mw_bins=mw_bins, seed=seed, logger=logger
    )
    
                             
    _del_data_attr(pyg_dataset, 'smiles')
    
                   
    trn = pyg_dataset[torch.LongTensor(trn_id)]
    val = pyg_dataset[torch.LongTensor(val_id)]
    test = pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

                              
    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'molecular_weight',
        'task_type': task_type,
        'mw_bins': mw_bins,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': len(trn_id),
        'val_size': len(val_id),
        'test_size': len(test_id)
    }
    
    metadata = get_cache_metadata(cache_key, dataset_config)
    
                                       
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_cache_with_metadata(save_path, trn, val, test, metadata)
    
    process_time = time.time() - start_time
    
    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)
    
                       
                       
    return load_dataset_molecular_weight(path, dataset, seed, task_type, tasks, mw_bins, descriptor_types, use_pharmacophores, logger)


                                            
                            
                                            

def similarity_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1),
                     similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                     seed=1, logger=None):


    assert sum(sizes) == 1

    if logger:
        logger.info('Computing Morgan fingerprints for similarity-based split...')

    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    smiles_data = _get_data_attr(pyg_dataset, "smiles")

    fps = []
    for smi in smiles_data:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), 2, nBits=2048))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if logger:
        logger.info(f'Clustering {len(fps)} molecules with similarity threshold {similarity_threshold:.2f}...')

    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - sim for sim in sims])

    clusters = Butina.ClusterData(
        dists,
        len(fps),
        1 - similarity_threshold,
        isDistData=True
    )

    cluster_sets = [list(cluster) for cluster in clusters]
    random_gen = Random(seed)

    big_cluster_sets = []
    small_cluster_sets = []
    for cluster_set in cluster_sets:
        if len(cluster_set) > val_size / 2 or len(cluster_set) > test_size / 2:
            big_cluster_sets.append(cluster_set)
        else:
            small_cluster_sets.append(cluster_set)

    random_gen.shuffle(big_cluster_sets)
    random_gen.shuffle(small_cluster_sets)
    cluster_sets = big_cluster_sets + small_cluster_sets

    train_ids, val_ids, test_ids = [], [], []
    train_cluster_count, val_cluster_count, test_cluster_count = 0, 0, 0

    for cluster_set in cluster_sets:
        if len(train_ids) + len(cluster_set) <= train_size:
            train_ids += cluster_set
            train_cluster_count += 1
        elif len(val_ids) + len(cluster_set) <= val_size:
            val_ids += cluster_set
            val_cluster_count += 1
        else:
            test_ids += cluster_set
            test_cluster_count += 1

    if logger:
        cluster_sizes = [len(cluster) for cluster in cluster_sets]
        logger.info(f'Similarity split results:')
        logger.info(f'  Total similarity clusters: {len(cluster_sets)}')
        logger.info(f'  Average cluster size: {np.mean(cluster_sizes):.2f}')
        logger.info(f'  Largest cluster size: {np.max(cluster_sizes)}')
        logger.info(f'  Train clusters: {train_cluster_count} ({len(train_ids)} molecules)')
        logger.info(f'  Val clusters: {val_cluster_count} ({len(val_ids)} molecules)')
        logger.info(f'  Test clusters: {test_cluster_count} ({len(test_ids)} molecules)')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    if task_type == 'classification':
        weights = []
        y_data = _get_data_attr(pyg_dataset, "y")
        for i in range(len(tasks)):
            pos_len = (y_data[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    else:
        weights = None

    return train_ids, val_ids, test_ids, weights


def load_dataset_similarity(path, dataset, seed, task_type, tasks=None,
                            descriptor_types=None, use_pharmacophores=False,
                            logger=None, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):


    cache_key = get_cache_key(
        dataset, seed, 'similarity', descriptor_types, use_pharmacophores,
        similarity_threshold=similarity_threshold
    )
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')

    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Similarity Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")

        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time

        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)

        return trn, val, test

    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Similarity Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")

    start_time = time.time()

    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )

    trn_id, val_id, test_id, weights = similarity_split(
        pyg_dataset, task_type=task_type, tasks=tasks,
        similarity_threshold=similarity_threshold, seed=seed, logger=logger
    )

    _del_data_attr(pyg_dataset, 'smiles')

    trn = pyg_dataset[torch.LongTensor(trn_id)]
    val = pyg_dataset[torch.LongTensor(val_id)]
    test = pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'similarity',
        'task_type': task_type,
        'similarity_threshold': similarity_threshold,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': len(trn_id),
        'val_size': len(val_id),
        'test_size': len(test_id)
    }

    metadata = get_cache_metadata(cache_key, dataset_config)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_cache_with_metadata(save_path, trn, val, test, metadata)

    process_time = time.time() - start_time

    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)

    return load_dataset_similarity(
        path, dataset, seed, task_type, tasks, descriptor_types,
        use_pharmacophores, logger, similarity_threshold
    )

def load_dataset_noise(path, dataset, seed, task_type, tasks, rate,
                      descriptor_types=None, use_pharmacophores=False, logger=None):

    
                                 
    cache_key = get_cache_key(dataset, seed, 'noise', descriptor_types, use_pharmacophores, rate)
    save_path = os.path.join(path, 'processed', f'train_valid_test_{cache_key}.ckpt')
    
                           
    if os.path.isfile(save_path):
        if logger:
            logger.info("=" * 60)
            logger.info("LOADING FROM CACHE (Noise Split)")
            logger.info("=" * 60)
            logger.info(f"✓ Cache found: {os.path.basename(save_path)}")
        
        start_time = time.time()
        trn, val, test, metadata = load_cache_with_validation(save_path, logger)
        load_time = time.time() - start_time
        
        if logger:
            logger.info(f"✓ Loaded train/val/test splits in {load_time:.2f}s")
            logger.info(f"  Train: {len(trn)} samples")
            logger.info(f"  Val: {len(val)} samples")
            logger.info(f"  Test: {len(test)} samples")
            logger.info("=" * 60)
        
        return trn, val, test
    
                                          
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING NEW DATASET (Noise Split)")
        logger.info("=" * 60)
        logger.info(f"× No cache found, processing data...")
    
    start_time = time.time()
    
                    
    pyg_dataset = EnhancedMolDataset(
        root=path, dataset=dataset, task_type=task_type,
        tasks=tasks, descriptor_types=descriptor_types,
        use_pharmacophores=use_pharmacophores, logger=logger
    )
    
                             
    _del_data_attr(pyg_dataset, 'smiles')

                         
    train_size = int(0.8 * len(pyg_dataset))
    val_size = int(0.1 * len(pyg_dataset))
    test_size = len(pyg_dataset) - train_size - val_size

    pyg_dataset, perm = pyg_dataset.shuffle(return_perm=True)
    trn_perm, val_perm = perm[:train_size], perm[train_size:(train_size + val_size)]
    trn_cutoff, val_cutoff = int(train_size * rate), int(val_size * rate)
    trn_noise_perm, val_noise_perm = trn_perm[:trn_cutoff], val_perm[:val_cutoff]
    noise_perm = torch.cat([trn_noise_perm, val_noise_perm])

                                   
    y_data = _get_data_attr(pyg_dataset, "y")
    y_data[noise_perm] = 1 - y_data[noise_perm]
    _set_data_attr(pyg_dataset, "y", y_data)

    trn = pyg_dataset[:train_size]
    val = pyg_dataset[train_size:(train_size + val_size)]
    test = pyg_dataset[(train_size + val_size):]

    if logger:
        logger.info(f'Dataset split with {rate*100:.0f}% noise:')
        logger.info(f'  Total: {len(pyg_dataset)} molecules')
        logger.info(f'  Train: {train_size} (noisy: {trn_cutoff})')
        logger.info(f'  Val: {val_size} (noisy: {val_cutoff})')
        logger.info(f'  Test: {test_size}')

                       
    weights = []
    pos_len = (y_data.sum()).item()
    neg_len = len(pyg_dataset) - pos_len
    weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    trn.weights = weights
    
    if logger:
        logger.info(f'Class weights: {weights}')

                              
    dataset_config = {
        'dataset': dataset,
        'seed': seed,
        'split_type': 'noise',
        'task_type': task_type,
        'noise_rate': rate,
        'descriptor_types': descriptor_types,
        'use_pharmacophores': use_pharmacophores,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }
    
    metadata = get_cache_metadata(cache_key, dataset_config)
    
                                       
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_cache_with_metadata(save_path, trn, val, test, metadata)
    
    process_time = time.time() - start_time
    
    if logger:
        logger.info(f"✓ Saved cache to {os.path.basename(save_path)}")
        logger.info(f"✓ Total time: {process_time:.2f}s")
        logger.info("=" * 60)
    
                       
    return load_dataset_noise(path, dataset, seed, task_type, tasks, rate, descriptor_types, use_pharmacophores, logger)


                                            
                  
                                            

def build_dataset(cfg, logger):

    cfg.defrost()
    task_name = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))
    if cfg.DATA.TASK_TYPE == 'classification':
        out_dim = 2 * len(task_name)
    elif cfg.DATA.TASK_TYPE == 'regression':
        out_dim = len(task_name)
    else:
        raise Exception('Unknown task type')
    
    descriptor_types = []
    if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE:
        descriptor_types = cfg.DESCRIPTOR.TYPES
    
    use_pharmacophores = False
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'THREE_LEVEL'):
        use_pharmacophores = cfg.MODEL.THREE_LEVEL.ENABLE
    
    opts = ['DATA.TASK_NAME', task_name, 'MODEL.OUT_DIM', out_dim]
    cfg.defrost()
    cfg.merge_from_list(opts)
    cfg.freeze()

    if cfg.DATA.SPLIT_TYPE == 'random':
        train_dataset, valid_dataset, test_dataset = load_dataset_random(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME,
            descriptor_types, use_pharmacophores, logger
        )
    elif cfg.DATA.SPLIT_TYPE == 'scaffold':
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME,
            descriptor_types, use_pharmacophores, logger
        )
    elif cfg.DATA.SPLIT_TYPE == 'scaffoldrandom':
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffoldrandom(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME,
            descriptor_types, use_pharmacophores, logger
        )
    elif cfg.DATA.SPLIT_TYPE == 'molecular_weight':
        mw_bins = cfg.DATA.MW_BINS if hasattr(cfg.DATA, 'MW_BINS') and cfg.DATA.MW_BINS else None
        train_dataset, valid_dataset, test_dataset = load_dataset_molecular_weight(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME,
            mw_bins, descriptor_types, use_pharmacophores, logger
        )
    elif cfg.DATA.SPLIT_TYPE == 'similarity':
        train_dataset, valid_dataset, test_dataset = load_dataset_similarity(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME,
            descriptor_types, use_pharmacophores, logger
        )
    elif cfg.DATA.SPLIT_TYPE == 'noise':
        train_dataset, valid_dataset, test_dataset = load_dataset_noise(
            cfg.DATA.DATA_PATH, cfg.DATA.DATASET, cfg.SEED,
            cfg.DATA.TASK_TYPE, cfg.DATA.TASK_NAME, cfg.DATA.RATE,
            descriptor_types, use_pharmacophores, logger
        )
    else:
        raise Exception('Unknown dataset split type')

    return train_dataset, valid_dataset, test_dataset


def build_loader(cfg, logger):

    train_dataset, valid_dataset, test_dataset = build_dataset(cfg, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    weights = train_dataset.weights

    return train_dataloader, valid_dataloader, test_dataloader, weights