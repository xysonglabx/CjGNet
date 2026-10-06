                       


import torch
from torch import nn
import torch.nn.functional as F


                                         
                                       
                                         
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self, out0, out1):
        device = out0.device
        batch_size, _ = out0.shape

                                          
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

                                                   
        output = torch.cat([out0, out1], 0)

                                                                         
        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature
                                                                      
        logits = logits[~torch.eye(2 * batch_size, dtype=torch.bool, device=out0.device)].view(2 * batch_size, -1)

                                                                                
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels])

        loss = self.cross_entropy(logits, labels)

        return loss


class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, out0, out1):
        loss = self.mse(out0, out1)
        return loss


                                         
                                                    
                                         
class HierarchicalContrastiveLoss(nn.Module):


    def __init__(self, temperature=0.1, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.temperature = temperature
                                           
        self.alpha = alpha                             
        self.beta = beta                                     
        self.gamma = gamma                      
        
        self.ntxent = NTXentLoss(temperature)
        
    def forward(self, atom_features, pharm_features, mol_features):


        losses = {}
        
                                             
        if atom_features is not None and pharm_features is not None:
            losses['atom_pharm'] = self.ntxent(atom_features, pharm_features)
        else:
            losses['atom_pharm'] = torch.tensor(0.0, device=mol_features.device)
        
                                                 
        if pharm_features is not None and mol_features is not None:
            losses['pharm_mol'] = self.ntxent(pharm_features, mol_features)
        else:
            losses['pharm_mol'] = torch.tensor(0.0, device=mol_features.device)
        
                                                      
        if atom_features is not None and mol_features is not None:
            losses['cross_scale'] = self.ntxent(atom_features, mol_features)
        else:
            losses['cross_scale'] = torch.tensor(0.0, device=mol_features.device)
        
                              
        total_loss = (self.alpha * losses['atom_pharm'] + 
                     self.beta * losses['pharm_mol'] + 
                     self.gamma * losses['cross_scale'])
        
        losses['total'] = total_loss
        return losses


                                         
                                 
                                         
class CrossScaleAlignmentLoss(nn.Module):


    def __init__(self, hidden_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
                                               
        self.atom_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pharm_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mol_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, atom_feat, pharm_feat, mol_feat):


                          
        atom_proj = self.atom_proj(atom_feat)
        pharm_proj = self.pharm_proj(pharm_feat)
        mol_proj = self.mol_proj(mol_feat)
        
                   
        atom_proj = F.normalize(atom_proj, dim=1)
        pharm_proj = F.normalize(pharm_proj, dim=1)
        mol_proj = F.normalize(mol_proj, dim=1)
        
                                       
        sim_atom_pharm = self.cosine_sim(atom_proj, pharm_proj)
        sim_pharm_mol = self.cosine_sim(pharm_proj, mol_proj)
        sim_atom_mol = self.cosine_sim(atom_proj, mol_proj)
        
                                               
        prob_atom_pharm = torch.sigmoid(sim_atom_pharm / self.temperature)
        prob_pharm_mol = torch.sigmoid(sim_pharm_mol / self.temperature)
        prob_atom_mol = torch.sigmoid(sim_atom_mol / self.temperature)
        
                                                                             
        transitivity_loss = F.mse_loss(
            prob_atom_mol,
            prob_atom_pharm * prob_pharm_mol
        )
        
                                 
        alignment_loss = -torch.mean(prob_atom_pharm + prob_pharm_mol + prob_atom_mol)
        
        total_loss = alignment_loss + 0.5 * transitivity_loss
        
        return total_loss


                                         
                                            
                                         
class DescriptorReconstructionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DescriptorReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred_descriptors, true_descriptors):
        if pred_descriptors is None or true_descriptors is None:
            return torch.tensor(0.0, device=pred_descriptors.device if pred_descriptors is not None else 'cpu')
        
        return self.mse(pred_descriptors, true_descriptors)


                                         
                                          
                                         
class ThreeLevelMultiTaskLoss(nn.Module):


    def __init__(self, main_loss, hierarchical_cl_loss=None, cross_scale_loss=None,
                 descriptor_loss=None, lambda_main=1.0, lambda_hierarchical_cl=0.2,
                 lambda_cross_scale=0.1, lambda_descriptor=0.3):
        super().__init__()
        
        self.main_loss = main_loss
        self.hierarchical_cl_loss = hierarchical_cl_loss
        self.cross_scale_loss = cross_scale_loss
        self.descriptor_loss = descriptor_loss
        
                      
        self.lambda_main = lambda_main
        self.lambda_hierarchical_cl = lambda_hierarchical_cl
        self.lambda_cross_scale = lambda_cross_scale
        self.lambda_descriptor = lambda_descriptor
        
    def forward(self, model_output, targets, descriptors=None):


        losses = {}
        
                                        
        if isinstance(model_output, dict):
            main_pred = model_output.get('main')
            atom_vec = model_output.get('atom_vec')
                                                            
            pharm_vec = model_output.get('pharm_vec')
            if pharm_vec is None:
                pharm_vec = model_output.get('fra_vec')
            mol_vec = model_output.get('mol_vec')
            pred_descriptors = model_output.get('descriptors')
        else:
                                    
            if isinstance(model_output, tuple) and len(model_output) == 3:
                main_pred, mol_vec, pharm_vec = model_output
                atom_vec = None
                pred_descriptors = None
            else:
                main_pred = model_output
                atom_vec = pharm_vec = mol_vec = pred_descriptors = None
        
                           
        if main_pred is not None:
            losses['main'] = self.main_loss(main_pred, targets)
        else:
            losses['main'] = torch.tensor(0.0, device=targets.device)
        
                                          
        if self.hierarchical_cl_loss is not None and pharm_vec is not None:
            if atom_vec is not None and mol_vec is not None:
                                         
                cl_losses = self.hierarchical_cl_loss(atom_vec, pharm_vec, mol_vec)
                losses['hierarchical_cl'] = cl_losses['total']
                losses['cl_atom_pharm'] = cl_losses.get('atom_pharm', 0.0)
                losses['cl_pharm_mol'] = cl_losses.get('pharm_mol', 0.0)
                losses['cl_cross_scale'] = cl_losses.get('cross_scale', 0.0)
            elif mol_vec is not None:
                                                     
                ntxent = NTXentLoss(temperature=0.1)
                losses['hierarchical_cl'] = ntxent(pharm_vec, mol_vec)
            else:
                losses['hierarchical_cl'] = torch.tensor(0.0, device=targets.device)
        else:
            losses['hierarchical_cl'] = torch.tensor(0.0, device=targets.device)
        
                                       
        if self.cross_scale_loss is not None and all(v is not None for v in [atom_vec, pharm_vec, mol_vec]):
            losses['cross_scale'] = self.cross_scale_loss(atom_vec, pharm_vec, mol_vec)
        else:
            losses['cross_scale'] = torch.tensor(0.0, device=targets.device)
        
                                           
        if self.descriptor_loss is not None and pred_descriptors is not None and descriptors is not None:
            losses['descriptor'] = self.descriptor_loss(pred_descriptors, descriptors)
        else:
            losses['descriptor'] = torch.tensor(0.0, device=targets.device)
        
                                     
        total_loss = (
            self.lambda_main * losses['main'] +
            self.lambda_hierarchical_cl * losses['hierarchical_cl'] +
            self.lambda_cross_scale * losses['cross_scale'] +
            self.lambda_descriptor * losses['descriptor']
        )
        
        losses['total'] = total_loss
        
        return losses


                                         
                        
                                         
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        target = target.float()
        pt = torch.softmax(inputs, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


                                         
                                                 
                                         
class JointLoss(nn.Module):
    def __init__(self, loss, cl_loss=None, alpha=0.5):
        super(JointLoss, self).__init__()
        self.loss = loss
        self.cl_loss = cl_loss
        self.alpha = alpha

    def forward(self, output, target, vec0=None, vec1=None):
        if self.cl_loss is None:
            loss = self.loss(output, target)
        else:
            loss = self.alpha * self.cl_loss(vec0, vec1) + (1 - self.alpha) * self.loss(output, target)
        return loss


                                         
                                                        
                                         
class MultiTaskLoss(nn.Module):
    def __init__(self, main_loss, cl_loss=None, descriptor_loss=None, 
                 lambda_main=1.0, lambda_contrastive=0.1, lambda_descriptor=0.5):
        super(MultiTaskLoss, self).__init__()
        
        self.main_loss = main_loss
        self.cl_loss = cl_loss
        self.descriptor_loss = descriptor_loss
        
        self.lambda_main = lambda_main
        self.lambda_contrastive = lambda_contrastive
        self.lambda_descriptor = lambda_descriptor

    def forward(self, model_output, targets, descriptors=None):
        losses = {}
        
        if isinstance(model_output, dict):
            main_pred = model_output.get('main')
            mol_vec = model_output.get('mol_vec')
            fra_vec = model_output.get('fra_vec')
            pred_descriptors = model_output.get('descriptors')
        elif isinstance(model_output, tuple) and len(model_output) == 3:
            main_pred, mol_vec, fra_vec = model_output
            pred_descriptors = None
        else:
            main_pred = model_output
            mol_vec = fra_vec = pred_descriptors = None
        
        if main_pred is not None:
            losses['main'] = self.main_loss(main_pred, targets)
        else:
            losses['main'] = torch.tensor(0.0, device=targets.device)
        
        if self.cl_loss is not None and mol_vec is not None and fra_vec is not None:
            losses['contrastive'] = self.cl_loss(mol_vec, fra_vec)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=targets.device)
        
        if self.descriptor_loss is not None and pred_descriptors is not None and descriptors is not None:
            losses['descriptor'] = self.descriptor_loss(pred_descriptors, descriptors)
        else:
            losses['descriptor'] = torch.tensor(0.0, device=targets.device)
        
        total_loss = (
            self.lambda_main * losses['main'] +
            self.lambda_contrastive * losses['contrastive'] +
            self.lambda_descriptor * losses['descriptor']
        )
        
        losses['total'] = total_loss
        
        if isinstance(model_output, dict):
            return losses
        else:
            return total_loss


                                         
                            
                                         
def bulid_loss(cfg, weight=None):


                          
    if cfg.DATA.TASK_TYPE == 'classification':
        if weight is not None:
            main_loss = nn.CrossEntropyLoss(weight=weight) if not cfg.LOSS.FL_LOSS else FocalLoss(alpha=1/weight[0])
        else:
            main_loss = nn.CrossEntropyLoss()
    else:
                                                  
        use_huber = getattr(cfg.LOSS, 'USE_HUBER', False)
        if use_huber:
            huber_delta = getattr(cfg.LOSS, 'HUBER_DELTA', 1.0)
            main_loss = nn.HuberLoss(delta=huber_delta)
        else:
            main_loss = nn.MSELoss()
    
                                               
    use_three_level = False
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'THREE_LEVEL'):
        use_three_level = cfg.MODEL.THREE_LEVEL.ENABLE
    
                                         
    if use_three_level:
                                      
        hierarchical_cl_loss = None
        cross_scale_loss = None
        
        if cfg.LOSS.CL_LOSS and hasattr(cfg.LOSS, 'THREE_LEVEL'):
            if cfg.LOSS.THREE_LEVEL.HIERARCHICAL_CL:
                hierarchical_cl_loss = HierarchicalContrastiveLoss(
                    temperature=cfg.LOSS.TEMPERATURE,
                    alpha=0.4,                     
                    beta=0.4,                     
                    gamma=0.2                       
                )
            
            if cfg.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE > 0:
                cross_scale_loss = CrossScaleAlignmentLoss(
                    hidden_dim=cfg.MODEL.HID,
                    temperature=cfg.LOSS.TEMPERATURE
                )
        
                         
        descriptor_loss = DescriptorReconstructionLoss() if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE else None
        
                          
        lambda_main = cfg.LOSS.MULTITASK.LAMBDA_MAIN if hasattr(cfg.LOSS, 'MULTITASK') else 1.0
        lambda_hierarchical_cl = cfg.LOSS.THREE_LEVEL.LAMBDA_PHARM_CONTRASTIVE if hasattr(cfg.LOSS, 'THREE_LEVEL') else 0.2
        lambda_cross_scale = cfg.LOSS.THREE_LEVEL.LAMBDA_CROSS_SCALE if hasattr(cfg.LOSS, 'THREE_LEVEL') else 0.1
        lambda_descriptor = cfg.LOSS.MULTITASK.LAMBDA_DESCRIPTOR if hasattr(cfg.LOSS, 'MULTITASK') else 0.3
        
                                            
        loss_fn = ThreeLevelMultiTaskLoss(
            main_loss=main_loss,
            hierarchical_cl_loss=hierarchical_cl_loss,
            cross_scale_loss=cross_scale_loss,
            descriptor_loss=descriptor_loss,
            lambda_main=lambda_main,
            lambda_hierarchical_cl=lambda_hierarchical_cl,
            lambda_cross_scale=lambda_cross_scale,
            lambda_descriptor=lambda_descriptor
        )
        
                                                                       
        class ThreeLevelWrapper:
            def __init__(self, loss_fn):
                self.loss_fn = loss_fn
                self.return_dict = False                                               
            
            def __call__(self, model_output, target, vec0=None, vec1=None, vec2=None, descriptors=None):
                                           
                if not isinstance(model_output, dict) and vec0 is not None:
                                         
                    model_output = {
                        'main': model_output,
                        'atom_vec': vec2 if vec2 is not None else None,
                        'pharm_vec': vec0,
                        'mol_vec': vec1,
                        'descriptors': None
                    }
                
                result = self.loss_fn(model_output, target, descriptors)
                
                                                                            
                                                                          
                if isinstance(result, dict):
                    return result['total']                           
                else:
                    return result
            
            def get_detailed_loss(self, model_output, target, descriptors=None):

                return self.loss_fn(model_output, target, descriptors)
        
        return ThreeLevelWrapper(loss_fn)
    
    else:
                                            
        cl_loss = NTXentLoss(temperature=cfg.LOSS.TEMPERATURE) if cfg.LOSS.CL_LOSS else None
        descriptor_loss = DescriptorReconstructionLoss() if hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE else None
        
        lambda_main = 1.0
        lambda_contrastive = cfg.LOSS.ALPHA
        lambda_descriptor = 0.5
        
        if hasattr(cfg, 'LOSS') and hasattr(cfg.LOSS, 'MULTITASK'):
            lambda_main = cfg.LOSS.MULTITASK.LAMBDA_MAIN
            lambda_contrastive = cfg.LOSS.MULTITASK.LAMBDA_CONTRASTIVE
            lambda_descriptor = cfg.LOSS.MULTITASK.LAMBDA_DESCRIPTOR
        
        use_multitask = (hasattr(cfg, 'DESCRIPTOR') and cfg.DESCRIPTOR.ENABLE) or cfg.LOSS.CL_LOSS
        
        if use_multitask:
            multi_task_loss = MultiTaskLoss(
                main_loss=main_loss,
                cl_loss=cl_loss,
                descriptor_loss=descriptor_loss,
                lambda_main=lambda_main,
                lambda_contrastive=lambda_contrastive,
                lambda_descriptor=lambda_descriptor
            )
            
            class MultiTaskWrapper:
                def __init__(self, multi_task_loss):
                    self.multi_task_loss = multi_task_loss
                
                def __call__(self, model_output, target, vec0=None, vec1=None, descriptors=None):
                    if not isinstance(model_output, dict) and (vec0 is not None or vec1 is not None):
                        if vec0 is not None and vec1 is not None:
                            model_output = (model_output, vec0, vec1)
                    
                    result = self.multi_task_loss(model_output, target, descriptors)
                                                     
                    if isinstance(result, dict):
                        return result['total']
                    return result
            
            return MultiTaskWrapper(multi_task_loss)
        else:
            joint_loss = JointLoss(loss=main_loss, cl_loss=cl_loss, alpha=cfg.LOSS.ALPHA)
            return joint_loss