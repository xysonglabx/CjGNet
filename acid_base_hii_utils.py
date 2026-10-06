                       


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

HALOGENS = {"F", "Cl", "Br", "I"}
HALOGEN_ORDER = ["F", "Cl", "Br", "I"]

                                                                           
                                                                                      
HALOGEN_PARAMS: Dict[str, Dict[str, float]] = {
    "F":  {"sigma_ind": 1.00, "sigma_res": 0.42, "polar": 0.20, "size": 0.25},
    "Cl": {"sigma_ind": 0.72, "sigma_res": 0.33, "polar": 0.55, "size": 0.55},
    "Br": {"sigma_ind": 0.64, "sigma_res": 0.28, "polar": 0.72, "size": 0.72},
    "I":  {"sigma_ind": 0.55, "sigma_res": 0.20, "polar": 1.00, "size": 1.00},
}

POSITION_FACTOR = {
    "ortho": 1.00,
    "meta": 0.65,
    "para": 0.85,
    "same_ring_other": 0.55,
    "same_aromatic_system": 0.50,
    "aliphatic": 0.25,
    "unknown": 0.20,
}

RESONANCE_POSITION_FACTOR = {
    "ortho": 0.85,
    "meta": 0.15,
    "para": 1.00,
    "same_ring_other": 0.45,
    "same_aromatic_system": 0.35,
    "aliphatic": 0.05,
    "unknown": 0.05,
}

RING_COMMUNICATION_FACTOR = {
    "benzene": 1.00,
    "naphthalene_or_fused_carbocycle": 1.12,
    "pyridine": 1.40,
    "diazine_or_polyaza_aromatic": 1.55,
    "five_member_heteroaromatic": 1.35,
    "fused_heteroaromatic": 1.45,
    "other_aromatic": 1.10,
    "aliphatic": 0.35,
    "unknown": 0.50,
}

                                                                          
                                                                              
ACID_CENTER_RESPONSE = {
    "carboxylic_acid":      {"ind": 1.05, "res": 0.95, "aro": 1.00, "steric": 1.00},
    "phenol":              {"ind": 0.95, "res": 1.25, "aro": 1.25, "steric": 1.25},
    "sulfonamide":         {"ind": 1.20, "res": 1.15, "aro": 1.15, "steric": 1.45},
    "sulfonic_acid":       {"ind": 0.95, "res": 0.80, "aro": 0.85, "steric": 0.90},
    "thiol":               {"ind": 0.95, "res": 0.80, "aro": 0.90, "steric": 1.10},
    "imide":               {"ind": 1.10, "res": 1.15, "aro": 1.05, "steric": 1.25},
    "azole_NH":            {"ind": 1.10, "res": 1.25, "aro": 1.30, "steric": 1.35},
    "generic_OH":          {"ind": 0.80, "res": 0.80, "aro": 0.80, "steric": 0.85},
    "generic_NH":          {"ind": 0.85, "res": 0.80, "aro": 0.80, "steric": 0.90},
    "unknown_acid":        {"ind": 0.75, "res": 0.75, "aro": 0.75, "steric": 0.75},
}

                                                                                     
                                                                                            
                                                                                   
ACID_TOTAL_WEIGHTS = {
    "inductive": 1.00,
    "resonance": -0.22,
    "aromatic": 0.16,
    "steric": 0.28,
}


def acid_total_v2(
    inductive: float,
    resonance: float,
    aromatic: float,
    steric: float,
    floor_zero: bool = False,
) -> float:


    total = (
        ACID_TOTAL_WEIGHTS["inductive"] * float(inductive)
        + ACID_TOTAL_WEIGHTS["resonance"] * float(resonance)
        + ACID_TOTAL_WEIGHTS["aromatic"] * float(aromatic)
        + ACID_TOTAL_WEIGHTS["steric"] * float(steric)
    )
    return max(0.0, total) if floor_zero else float(total)


BASE_CENTER_RESPONSE = {
                                                      
                                                                           
                                                                     
                                                       
                                                                       
                                                                     
    "aliphatic_amine":      {"ind": 1.00, "lp": 1.00, "arN": 0.15, "res": 0.20, "field": 0.75},
    "aniline_N":            {"ind": 1.10, "lp": 0.75, "arN": 0.85, "res": 1.10, "field": 0.85},
    "pyridine_N":           {"ind": 1.45, "lp": 1.15, "arN": 1.50, "res": 1.25, "field": 1.00},
    "diazine_N":            {"ind": 1.55, "lp": 1.10, "arN": 1.70, "res": 1.35, "field": 1.05},
    "azole_pyridine_N":     {"ind": 1.35, "lp": 1.10, "arN": 1.35, "res": 1.20, "field": 1.05},
    "imidazole_pyridine_N": {"ind": 1.40, "lp": 1.10, "arN": 1.50, "res": 1.30, "field": 1.05},
    "amidine":              {"ind": 0.85, "lp": 1.00, "arN": 0.35, "res": 1.05, "field": 0.95},
    "guanidine":            {"ind": 0.75, "lp": 1.00, "arN": 0.30, "res": 1.10, "field": 0.95},
    "generic_basic_N":      {"ind": 0.95, "lp": 0.85, "arN": 0.65, "res": 0.70, "field": 0.85},
    "unknown_base":         {"ind": 0.80, "lp": 0.70, "arN": 0.55, "res": 0.60, "field": 0.75},
}

                                                                    
BASE_POSITION_FACTOR = {
    "ortho": 1.20,
    "meta": 0.55,
    "para": 1.00,
    "same_ring_other": 0.65,
    "same_aromatic_system": 0.60,
    "aliphatic": 0.05,
    "unknown": 0.05,
}

BASE_RESONANCE_POSITION_FACTOR = {
    "ortho": 0.80,
    "meta": 0.20,
    "para": 1.00,
    "same_ring_other": 0.45,
    "same_aromatic_system": 0.35,
    "aliphatic": 0.05,
    "unknown": 0.05,
}

BASE_RING_COMMUNICATION_FACTOR = {
    "benzene": 0.85,
    "naphthalene_or_fused_carbocycle": 0.95,
    "pyridine": 1.50,
    "diazine_or_polyaza_aromatic": 1.70,
    "five_member_heteroaromatic": 1.35,
    "fused_heteroaromatic": 1.50,
    "other_aromatic": 1.00,
    "aliphatic": 0.10,
    "unknown": 0.30,
}

                                                                           
                                                                   
ACID_CENTER_PRIOR = {
    "carboxylic_acid": 1.60,
    "sulfonic_acid": 1.60,
    "sulfonamide": 1.35,
    "imide": 1.35,
    "phenol": 1.25,
    "azole_NH": 1.25,
    "thiol": 1.10,
    "generic_OH": 0.65,
    "generic_NH": 0.50,
    "unknown_acid": 0.40,
}

BASE_CENTER_PRIOR = {
    "pyridine_N": 1.65,
    "diazine_N": 1.70,
    "imidazole_pyridine_N": 1.55,
    "azole_pyridine_N": 1.45,
    "aliphatic_amine": 1.40,
    "aniline_N": 1.10,
    "amidine": 1.25,
    "guanidine": 1.25,
    "generic_basic_N": 0.65,
    "unknown_base": 0.40,
}

ACID_TYPICAL_PKA = {
    "sulfonic_acid": (-0.5, 2.0),
    "carboxylic_acid": (4.5, 1.5),
    "sulfonamide": (7.5, 2.0),
    "phenol": (10.0, 2.0),
    "imide": (9.0, 2.0),
    "azole_NH": (12.0, 3.0),
    "thiol": (10.0, 2.5),
    "generic_OH": (12.0, 3.0),
    "generic_NH": (14.0, 4.0),
    "unknown_acid": (7.0, 5.0),
}

BASE_TYPICAL_PKA = {
    "aliphatic_amine": (9.8, 2.0),
    "aniline_N": (4.8, 1.8),
    "pyridine_N": (5.2, 2.0),
    "diazine_N": (2.5, 2.0),
    "azole_pyridine_N": (4.5, 2.5),
    "imidazole_pyridine_N": (6.5, 2.5),
    "amidine": (12.0, 2.0),
    "guanidine": (13.0, 2.0),
    "generic_basic_N": (7.0, 4.0),
    "unknown_base": (7.0, 5.0),
}


@dataclass(frozen=True)
class IonizableCenter:
    family: str                   
    center_type: str
    center_idx: int                                                      
    anchor_idx: int                                                               
    smarts_name: str = ""


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _unique_centers(centers: Iterable[IonizableCenter]) -> List[IonizableCenter]:
    seen = set()
    out = []
    priority = {
        "carboxylic_acid": 0,
        "sulfonic_acid": 0,
        "sulfonamide": 1,
        "imide": 1,
        "phenol": 1,
        "azole_NH": 1,
        "thiol": 2,
        "generic_OH": 9,
        "generic_NH": 9,
        "pyridine_N": 0,
        "diazine_N": 0,
        "azole_pyridine_N": 0,
        "imidazole_pyridine_N": 0,
        "amidine": 0,
        "guanidine": 0,
        "aniline_N": 1,
        "aliphatic_amine": 1,
        "generic_basic_N": 9,
    }
    for c in sorted(list(centers), key=lambda z: priority.get(z.center_type, 5)):
        key = (c.family, c.center_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def find_acidic_centers(mol: Chem.Mol) -> List[IonizableCenter]:
    centers: List[IonizableCenter] = []

                                                                     
    patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            carbon = int(match[0])
            oh = int(match[2])
            centers.append(IonizableCenter("acidic", "carboxylic_acid", oh, carbon, "carboxylic_acid"))

                                                                       
    patt = Chem.MolFromSmarts("[OX2H1][c]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            o = int(match[0]); c = int(match[1])
            centers.append(IonizableCenter("acidic", "phenol", o, c, "phenol"))

                                                       
    patt = Chem.MolFromSmarts("[SX4](=O)(=O)([OX2H1])")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            s = int(match[0]); oh = int(match[3])
            centers.append(IonizableCenter("acidic", "sulfonic_acid", oh, s, "sulfonic_acid"))

                                              
    patt = Chem.MolFromSmarts("[NX3;H1,H2][SX4](=O)(=O)")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0]); s = int(match[1])
            centers.append(IonizableCenter("acidic", "sulfonamide", n, s, "sulfonamide"))

                                                 
    patt = Chem.MolFromSmarts("[NX3;H1]([CX3](=O))[CX3](=O)")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0])
            centers.append(IonizableCenter("acidic", "imide", n, n, "imide"))

                                
    patt = Chem.MolFromSmarts("[nH]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0])
            centers.append(IonizableCenter("acidic", "azole_NH", n, n, "azole_NH"))

            
    patt = Chem.MolFromSmarts("[SX2H1]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            s = int(match[0])
            centers.append(IonizableCenter("acidic", "thiol", s, s, "thiol"))

                                                                                                 
    patt = Chem.MolFromSmarts("[OX2H1]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            o = int(match[0])
            atom = mol.GetAtomWithIdx(o)
            nbrs = atom.GetNeighbors()
            anchor = int(nbrs[0].GetIdx()) if nbrs else o
            centers.append(IonizableCenter("acidic", "generic_OH", o, anchor, "generic_OH"))

                                                                    
                                                                            
                                                                        
                                                                    

    return _unique_centers(centers)


def find_basic_centers(mol: Chem.Mol) -> List[IonizableCenter]:
    centers: List[IonizableCenter] = []

                                                              
    patt = Chem.MolFromSmarts("[CX3](=[NX2,NX3])[NX3]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            c = int(match[0]); n1 = int(match[1]); n2 = int(match[2])
            center_type = "amidine"
                                                                       
            carbon = mol.GetAtomWithIdx(c)
            n_count = sum(1 for nb in carbon.GetNeighbors() if nb.GetSymbol() == "N")
            if n_count >= 3:
                center_type = "guanidine"
            centers.append(IonizableCenter("basic", center_type, n1, c, center_type))
            centers.append(IonizableCenter("basic", center_type, n2, c, center_type))

                                                                                         
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        idx = atom.GetIdx()
        if atom.GetFormalCharge() > 0:
            continue
        if atom.GetIsAromatic() and atom.GetTotalNumHs() == 0:
            ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings() if idx in r]
            hetero_in_rings = []
            for ring in mol.GetRingInfo().AtomRings():
                if idx in ring:
                    hetero_in_rings.append(sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() not in {"C", "H"}))
            if ring_sizes and min(ring_sizes) == 5:
                ctype = "azole_pyridine_N"
                if any(h >= 2 for h in hetero_in_rings):
                    ctype = "imidazole_pyridine_N"
            elif any(h >= 2 for h in hetero_in_rings):
                ctype = "diazine_N"
            else:
                ctype = "pyridine_N"
            centers.append(IonizableCenter("basic", ctype, idx, idx, ctype))

                                                             
    patt = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-[CX3]=O);!$(N-[SX4](=O)(=O))][c]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0]); c = int(match[1])
            atom = mol.GetAtomWithIdx(n)
            if atom.GetFormalCharge() <= 0:
                centers.append(IonizableCenter("basic", "aniline_N", n, c, "aniline_N"))

                                                           
    patt = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$([nH]);!$(N-[CX3]=O);!$(N-[SX4](=O)(=O));!$(N=*)]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0])
            atom = mol.GetAtomWithIdx(n)
            has_aromatic_neighbor = any(nb.GetIsAromatic() for nb in atom.GetNeighbors())
            if atom.GetFormalCharge() <= 0 and not atom.GetIsAromatic() and not has_aromatic_neighbor:
                centers.append(IonizableCenter("basic", "aliphatic_amine", n, n, "aliphatic_amine"))

                                                              
    patt = Chem.MolFromSmarts("[N;!$([N+]);!$(N-[CX3]=O);!$(N-[SX4](=O)(=O));!$([nH])]")
    if patt is not None:
        for match in mol.GetSubstructMatches(patt):
            n = int(match[0])
            atom = mol.GetAtomWithIdx(n)
            if atom.GetFormalCharge() <= 0:
                centers.append(IonizableCenter("basic", "generic_basic_N", n, n, "generic_basic_N"))

    return _unique_centers(centers)


def find_ionizable_centers(mol: Chem.Mol) -> List[IonizableCenter]:
    return find_acidic_centers(mol) + find_basic_centers(mol)


def get_halogen_sites(mol: Chem.Mol) -> List[Dict[str, Any]]:
    sites = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in HALOGENS or atom.GetDegree() == 0:
            continue
        nbr = atom.GetNeighbors()[0]
        sites.append({
            "halogen": sym,
            "halogen_idx": int(atom.GetIdx()),
            "attached_idx": int(nbr.GetIdx()),
            "attached_symbol": nbr.GetSymbol(),
            "attached_is_aromatic": bool(nbr.GetIsAromatic()),
        })
    return sites


def classify_ring_type(mol: Chem.Mol, atom_idx: int) -> str:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    if not atom.GetIsAromatic():
        return "aliphatic"
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if atom_idx in r]
    if not rings:
        return "other_aromatic"

                                                   
    ring = sorted(rings, key=len)[0]
    size = len(ring)
    hetero = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() not in {"C", "H"})
    n_count = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "N")

    atom_ring_count = sum(1 for r in rings if atom_idx in r)
    fused = atom_ring_count > 1 or len(rings) > 1

    if size == 6 and hetero == 0:
        return "naphthalene_or_fused_carbocycle" if fused else "benzene"
    if size == 6 and n_count == 1 and hetero == 1:
        return "pyridine"
    if size == 6 and hetero >= 2:
        return "diazine_or_polyaza_aromatic"
    if size == 5 and hetero >= 1:
        return "fused_heteroaromatic" if fused else "five_member_heteroaromatic"
    if fused and hetero >= 1:
        return "fused_heteroaromatic"
    return "other_aromatic"


def _ring_distance_on_same_ring(mol: Chem.Mol, a: int, b: int) -> Optional[int]:
    for ring in mol.GetRingInfo().AtomRings():
        if a in ring and b in ring:
            ring_list = list(ring)
            n = len(ring_list)
            ia = ring_list.index(a)
            ib = ring_list.index(b)
            diff = abs(ia - ib)
            return int(min(diff, n - diff))
    return None


def classify_position(mol: Chem.Mol, hal_attached_idx: int, center_anchor_idx: int) -> str:
    hal_atom = mol.GetAtomWithIdx(int(hal_attached_idx))
    anchor_atom = mol.GetAtomWithIdx(int(center_anchor_idx))
    if not (hal_atom.GetIsAromatic() and anchor_atom.GetIsAromatic()):
        return "aliphatic"

    rd = _ring_distance_on_same_ring(mol, int(hal_attached_idx), int(center_anchor_idx))
    if rd is not None:
        if rd == 1:
            return "ortho"
        if rd == 2:
            return "meta"
        if rd == 3:
            return "para"
        return "same_ring_other"

                                                                                                   
    try:
        path = Chem.rdmolops.GetShortestPath(mol, int(hal_attached_idx), int(center_anchor_idx))
        if len(path) <= 7:
            return "same_aromatic_system"
    except Exception:
        pass
    return "unknown"


def graph_distance(mol: Chem.Mol, a: int, b: int, default: int = 6) -> int:
    try:
        path = Chem.rdmolops.GetShortestPath(mol, int(a), int(b))
        if path:
            return max(0, len(path) - 1)
    except Exception:
        pass
    return default


def distance_decay(d: float) -> float:
    return 1.0 / (1.0 + 0.45 * float(d))


def steric_geometry_factor(position: str, d: int) -> float:
    if position == "ortho":
        return 1.00
    if d <= 3:
        return 0.35
    return 0.12


                                             
                                                                             
                                                                              
                                                                                 
                                                                          
                                                                        
ALIPHATIC_ACID_TOTAL_WEIGHTS = {
    "inductive": 1.00,
    "field": 0.55,
    "proximity": 0.45,
    "steric": 0.25,
}


def aliphatic_field_geometry_factor(d: int) -> float:

    d = int(d)
    if d <= 2:
        return 1.00
    if d == 3:
        return 0.55
    if d == 4:
        return 0.25
    return 0.10


def aliphatic_proximity_factor(d: int) -> float:

    d = int(d)
    if d <= 2:
        return 1.00
    if d == 3:
        return 0.65
    if d == 4:
        return 0.35
    return 0.12


def aliphatic_steric_geometry_factor(d: int) -> float:

    d = int(d)
    if d <= 2:
        return 0.85
    if d == 3:
        return 0.45
    if d == 4:
        return 0.20
    return 0.08


def acid_aliphatic_total_v1(
    inductive: float,
    field: float,
    proximity: float,
    steric: float,
    floor_zero: bool = False,
) -> float:


    total = (
        ALIPHATIC_ACID_TOTAL_WEIGHTS["inductive"] * float(inductive)
        + ALIPHATIC_ACID_TOTAL_WEIGHTS["field"] * float(field)
        + ALIPHATIC_ACID_TOTAL_WEIGHTS["proximity"] * float(proximity)
        + ALIPHATIC_ACID_TOTAL_WEIGHTS["steric"] * float(steric)
    )
    return max(0.0, total) if floor_zero else float(total)


def base_field_geometry_factor(position: str, d: int) -> float:

    if position == "ortho":
        return 1.00
    if d <= 3:
        return 0.55
    return 0.15


def compute_pair_hii(mol: Chem.Mol, site: Dict[str, Any], center: IonizableCenter) -> Dict[str, Any]:
    halogen = site["halogen"]
    params = HALOGEN_PARAMS.get(halogen, HALOGEN_PARAMS["Cl"])
    attached_idx = int(site["attached_idx"])
    center_anchor = int(center.anchor_idx)
    center_idx = int(center.center_idx)

    d_anchor = graph_distance(mol, attached_idx, center_anchor, default=6)
    d_center = graph_distance(mol, attached_idx, center_idx, default=d_anchor)
    position = classify_position(mol, attached_idx, center_anchor)
    ring_type = classify_ring_type(mol, attached_idx)
    attached_is_aromatic = bool(site.get("attached_is_aromatic", False))

    p_factor = POSITION_FACTOR.get(position, POSITION_FACTOR["unknown"])
    r_factor = RESONANCE_POSITION_FACTOR.get(position, RESONANCE_POSITION_FACTOR["unknown"])
    ring_factor = RING_COMMUNICATION_FACTOR.get(ring_type, RING_COMMUNICATION_FACTOR["unknown"])
                                                                           
                                                                            
                                                                      
    q_res = 1.00
    q_aro = 1.00

                                                                                 
    if center.family == "acidic":
        resp = ACID_CENTER_RESPONSE.get(center.center_type, ACID_CENTER_RESPONSE["unknown_acid"])
        inductive = params["sigma_ind"] * distance_decay(d_center) * (0.85 + 0.15 * p_factor) * resp["ind"]
        resonance = params["sigma_res"] * r_factor * q_res * resp["res"]
        aromatic = params["polar"] * ring_factor * p_factor * q_aro * resp["aro"]
        steric = params["size"] * steric_geometry_factor(position, d_center) * resp["steric"]
        raw_total = inductive + resonance + aromatic + steric
        total = acid_total_v2(inductive, resonance, aromatic, steric)

                                                                      
                                                                               
                                                    
        aliphatic_inductive = 0.0
        aliphatic_field = 0.0
        aliphatic_proximity = 0.0
        aliphatic_steric = 0.0
        aliphatic_total = 0.0
        lonepair = 0.0
        aromatic_N = 0.0
        field = 0.0
    else:
                                                                             
                                                                   
                                                                              
                                                                  
                                                                       
                                                          
        resp = BASE_CENTER_RESPONSE.get(center.center_type, BASE_CENTER_RESPONSE["unknown_base"])
        base_pos = BASE_POSITION_FACTOR.get(position, BASE_POSITION_FACTOR["unknown"])
        base_res = BASE_RESONANCE_POSITION_FACTOR.get(position, BASE_RESONANCE_POSITION_FACTOR["unknown"])
        base_ring = BASE_RING_COMMUNICATION_FACTOR.get(ring_type, BASE_RING_COMMUNICATION_FACTOR["unknown"])
        q_base_aro = 1.00 if attached_is_aromatic else 0.05
        q_base_res = 1.00 if attached_is_aromatic else 0.08
        inductive = params["sigma_ind"] * distance_decay(d_center) * resp["ind"]
        lonepair = params["sigma_ind"] * distance_decay(d_center) * resp["lp"]
        aromatic_N = params["polar"] * base_ring * base_pos * q_base_aro * resp["arN"]
        resonance = params["sigma_res"] * base_res * q_base_res * resp["res"]
        field = params["polar"] * base_field_geometry_factor(position, d_center) * resp["field"]
                                                                                               
                                                             
        aromatic = aromatic_N
        steric = field
        total = inductive + lonepair + aromatic_N + resonance + field
        raw_total = total
        aliphatic_inductive = 0.0
        aliphatic_field = 0.0
        aliphatic_proximity = 0.0
        aliphatic_steric = 0.0
        aliphatic_total = 0.0

    return {
        "family": center.family,
        "center_type": center.center_type,
        "center_idx": center_idx,
        "center_anchor_idx": center_anchor,
        "halogen": halogen,
        "halogen_idx": int(site["halogen_idx"]),
        "attached_idx": attached_idx,
        "attached_is_aromatic": attached_is_aromatic,
        "position": position,
        "ring_type": ring_type,
        "graph_distance": int(d_center),
        "anchor_distance": int(d_anchor),
        "inductive": float(inductive),
        "resonance": float(resonance),
        "aromatic": float(aromatic),
        "steric": float(steric),
        "lonepair": float(lonepair),
        "aromatic_N": float(aromatic_N),
        "field": float(field),
        "aliphatic_inductive": float(aliphatic_inductive),
        "aliphatic_field": float(aliphatic_field),
        "aliphatic_proximity": float(aliphatic_proximity),
        "aliphatic_steric": float(aliphatic_steric),
        "aliphatic_total": float(aliphatic_total),
        "raw_total": float(raw_total),
        "total": float(total),
    }


def compute_acid_base_hii_pair_contributions(smiles: str) -> List[Dict[str, Any]]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    sites = get_halogen_sites(mol)
    centers = find_ionizable_centers(mol)
    rows: List[Dict[str, Any]] = []
    for site in sites:
        for center in centers:
            rows.append(compute_pair_hii(mol, site, center))
    return rows


def _pka_score(center_type: str, family: str, pka_value: Optional[float]) -> float:
    if pka_value is None or not np.isfinite(float(pka_value)):
        return 0.0
    table = ACID_TYPICAL_PKA if family == "acidic" else BASE_TYPICAL_PKA
    mean, width = table.get(center_type, table.get("unknown_acid" if family == "acidic" else "unknown_base"))
    width = max(float(width), 0.5)
    return -0.5 * ((float(pka_value) - float(mean)) / width) ** 2


def _center_prior(center_type: str, family: str) -> float:
    table = ACID_CENTER_PRIOR if family == "acidic" else BASE_CENTER_PRIOR
    return float(table.get(center_type, 0.4))


def _pair_weight_scores(pairs: List[Dict[str, Any]], family: str, pka_value: Optional[float] = None) -> np.ndarray:
    scores = []
    for p in pairs:
        d = float(p.get("graph_distance", 6.0))
        distance_score = 1.0 / (1.0 + d)
        score = (
            _center_prior(str(p.get("center_type", "unknown")), family)
            + 1.00 * distance_score
            + 0.60 * _pka_score(str(p.get("center_type", "unknown")), family, pka_value)
        )
        scores.append(score)
    return np.asarray(scores, dtype=float)


def _softmax(scores: np.ndarray, tau: float = 0.75) -> np.ndarray:
    if len(scores) == 0:
        return scores
    tau = max(float(tau), 1e-6)
    z = scores / tau
    z = z - np.nanmax(z)
    e = np.exp(z)
    if not np.isfinite(e).all() or e.sum() <= 0:
        return np.ones(len(scores), dtype=float) / float(len(scores))
    return e / e.sum()


def _components_for_family(family: str) -> List[str]:
    if family == "acidic":
        return ["inductive", "resonance", "aromatic", "steric", "raw_total", "total"]
    return ["inductive", "lonepair", "aromatic_N", "resonance", "field", "total"]


def _aggregate_pairs(
    pairs: List[Dict[str, Any]],
    family: str,
    aggregation: str = "target_weighted",
    pka_value: Optional[float] = None,
    weight_tau: float = 0.75,
) -> Dict[str, Any]:


    fam_pairs = [p for p in pairs if p.get("family") == family]
    prefix = "acid" if family == "acidic" else "base"
    comps = _components_for_family(family)
    out: Dict[str, Any] = {
        f"n_{prefix}_pairs": len(fam_pairs),
        f"{prefix}_center_type": "none",
        f"{prefix}_position": "none",
        f"{prefix}_ring_type": "none",
        f"{prefix}_min_distance": np.nan,
        f"{prefix}_weight_entropy": np.nan,
    }
                                                             
    for comp in [
        "inductive", "resonance", "aromatic", "steric", "lonepair", "aromatic_N",
        "field", "raw_total", "total",
        "aliphatic_inductive", "aliphatic_field", "aliphatic_proximity",
        "aliphatic_steric", "aliphatic_total",
    ]:
        for suffix in ["sum", "mean", "weighted", "selected"]:
            out[f"{prefix}_hii_{comp}_{suffix}"] = 0.0

    if not fam_pairs:
        return out

    scores = _pair_weight_scores(fam_pairs, family, pka_value=pka_value)
    weights = _softmax(scores, tau=weight_tau)
    entropy = -float(np.sum(weights * np.log(weights + 1e-12)))
    out[f"{prefix}_weight_entropy"] = entropy

                                                                                    
    top_idx = int(np.argmax(weights)) if len(weights) else 0
    top_pair = fam_pairs[top_idx]
    out[f"{prefix}_center_type"] = str(top_pair.get("center_type", "none"))
    out[f"{prefix}_position"] = str(top_pair.get("position", "none"))
    out[f"{prefix}_ring_type"] = str(top_pair.get("ring_type", "none"))
    distances = [float(p.get("graph_distance")) for p in fam_pairs if np.isfinite(float(p.get("graph_distance", np.nan)))]
    if distances:
        out[f"{prefix}_min_distance"] = float(np.min(distances))

    for comp in comps:
        vals = np.asarray([float(p.get(comp, 0.0)) for p in fam_pairs], dtype=float)
        sum_v = float(np.sum(vals))
        mean_v = float(np.mean(vals)) if len(vals) else 0.0
        weighted_v = float(np.sum(weights * vals)) if len(vals) else 0.0
        if aggregation == "sum":
            selected = sum_v
        elif aggregation == "pair_mean":
            selected = mean_v
        else:
            selected = weighted_v
        out[f"{prefix}_hii_{comp}_sum"] = sum_v
        out[f"{prefix}_hii_{comp}_mean"] = mean_v
        out[f"{prefix}_hii_{comp}_weighted"] = weighted_v
        out[f"{prefix}_hii_{comp}_selected"] = selected

                                                                                   
    if family == "basic":
        for suffix in ["sum", "mean", "weighted", "selected"]:
            out[f"{prefix}_hii_aromatic_{suffix}"] = out[f"{prefix}_hii_aromatic_N_{suffix}"]
            out[f"{prefix}_hii_steric_{suffix}"] = out[f"{prefix}_hii_field_{suffix}"]
    return out


def infer_pka_mode(
    smiles: str,
    pka_value: Optional[float] = None,
    pka_mode: Optional[str] = None,
    pka_type_value: Optional[Any] = None,
    acid_pair_count: int = 0,
    base_pair_count: int = 0,
) -> str:


    if pka_mode is not None and str(pka_mode).lower() in {"acidic", "basic", "ambiguous", "unknown"}:
        return str(pka_mode).lower()

    if pka_type_value is not None and not (isinstance(pka_type_value, float) and np.isnan(pka_type_value)):
        s = str(pka_type_value).strip().lower()
        if s in {"acid", "acidic", "acid_pka", "pka_acid", "a"}:
            return "acidic"
        if s in {"base", "basic", "pkah", "pka_h", "conjugate_acid", "base_pka", "basic_pka", "b"}:
            return "basic"
        if s in {"ambiguous", "mixed", "both"}:
            return "ambiguous"

    has_acid = acid_pair_count > 0
    has_base = base_pair_count > 0
    if has_acid and not has_base:
        return "acidic"
    if has_base and not has_acid:
        return "basic"
    if has_acid and has_base:
                                                                               
        if pka_value is not None and np.isfinite(float(pka_value)):
            v = float(pka_value)
                                                                                                                   
            if v <= 6.5:
                return "acidic"
            if v >= 9.5:
                return "basic"
        return "ambiguous"
    return "unknown"


def select_target_components(
    row: Dict[str, Any],
    pka_mode: str,
    aggregation: str = "target_weighted",
    ambiguous_policy: str = "mixed",
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:


    suffix = "selected"

    acid = {
        "inductive": float(row.get(f"acid_hii_inductive_{suffix}", 0.0)),
        "resonance": float(row.get(f"acid_hii_resonance_{suffix}", 0.0)),
        "aromatic": float(row.get(f"acid_hii_aromatic_{suffix}", 0.0)),
        "steric": float(row.get(f"acid_hii_steric_{suffix}", 0.0)),
        "lonepair": 0.0,
        "aromatic_N": 0.0,
        "field": 0.0,
        "aliphatic_total": 0.0,
        "total": float(row.get(f"acid_hii_total_{suffix}", 0.0)),
    }
    base = {
        "inductive": float(row.get(f"base_hii_inductive_{suffix}", 0.0)),
        "resonance": float(row.get(f"base_hii_resonance_{suffix}", 0.0)),
        "aromatic": 0.0,
        "steric": 0.0,
        "lonepair": float(row.get(f"base_hii_lonepair_{suffix}", 0.0)),
        "aromatic_N": float(row.get(f"base_hii_aromatic_N_{suffix}", 0.0)),
        "field": float(row.get(f"base_hii_field_{suffix}", 0.0)),
        "aliphatic_total": 0.0,
        "total": float(row.get(f"base_hii_total_{suffix}", 0.0)),
    }
    n_acid = int(row.get("n_acid_pairs", 0))
    n_base = int(row.get("n_base_pairs", 0))

    if pka_mode == "acidic" and n_acid > 0:
        return "acidic", acid, {"target_center_type": row.get("acid_center_type", "none")}
    if pka_mode == "basic" and n_base > 0:
        return "basic", base, {"target_center_type": row.get("base_center_type", "none")}

    if pka_mode == "ambiguous":
        if ambiguous_policy == "acid" and n_acid > 0:
            return "ambiguous_as_acidic", acid, {"target_center_type": row.get("acid_center_type", "none")}
        if ambiguous_policy == "base" and n_base > 0:
            return "ambiguous_as_basic", base, {"target_center_type": row.get("base_center_type", "none")}
        if ambiguous_policy == "max_hii":
            if acid["total"] >= base["total"] and n_acid > 0:
                return "ambiguous_max_acidic", acid, {"target_center_type": row.get("acid_center_type", "none")}
            if n_base > 0:
                return "ambiguous_max_basic", base, {"target_center_type": row.get("base_center_type", "none")}
        if n_acid > 0 and n_base > 0:
            mix = {k: 0.5 * (acid.get(k, 0.0) + base.get(k, 0.0)) for k in acid.keys()}
            return "ambiguous_mixed", mix, {"target_center_type": "acid_base_mixed"}

    if n_acid > 0 and n_base == 0:
        return "fallback_acidic", acid, {"target_center_type": row.get("acid_center_type", "none")}
    if n_base > 0 and n_acid == 0:
        return "fallback_basic", base, {"target_center_type": row.get("base_center_type", "none")}
    if n_acid > 0 and n_base > 0:
        mix = {k: 0.5 * (acid.get(k, 0.0) + base.get(k, 0.0)) for k in acid.keys()}
        return "fallback_mixed", mix, {"target_center_type": "acid_base_mixed"}

    empty = {"inductive": 0.0, "resonance": 0.0, "aromatic": 0.0, "steric": 0.0,
             "lonepair": 0.0, "aromatic_N": 0.0, "field": 0.0, "total": 0.0}
    return "unknown", empty, {"target_center_type": "none"}


def compute_site_acid_base_hii(
    smiles: str,
    halogen_idx: int,
    pka_value: Optional[float] = None,
    pka_mode: Optional[str] = None,
    pka_type_value: Optional[Any] = None,
    aggregation: str = "target_weighted",
    ambiguous_policy: str = "mixed",
    weight_tau: float = 0.75,
) -> Dict[str, Any]:

    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    sites = get_halogen_sites(mol)
    site = None
    for s in sites:
        if int(s["halogen_idx"]) == int(halogen_idx):
            site = s
            break
    if site is None:
        raise ValueError(f"Halogen index {halogen_idx} not found in {smiles}")

    centers = find_ionizable_centers(mol)
    pairs = [compute_pair_hii(mol, site, center) for center in centers]
    acid_agg = _aggregate_pairs(pairs, "acidic", aggregation=aggregation, pka_value=pka_value, weight_tau=weight_tau)
    base_agg = _aggregate_pairs(pairs, "basic", aggregation=aggregation, pka_value=pka_value, weight_tau=weight_tau)
    row: Dict[str, Any] = {}
    row.update(acid_agg)
    row.update(base_agg)

    inferred = infer_pka_mode(
        smiles=smiles,
        pka_value=pka_value,
        pka_mode=pka_mode,
        pka_type_value=pka_type_value,
        acid_pair_count=int(row.get("n_acid_pairs", 0)),
        base_pair_count=int(row.get("n_base_pairs", 0)),
    )
    selected_mode, target_comps, target_meta = select_target_components(
        row,
        inferred,
        aggregation=aggregation,
        ambiguous_policy=ambiguous_policy,
    )

    row.update({
        "pka_mode_inferred": inferred,
        "target_hii_mode": selected_mode,
        "hii_aggregation": aggregation,
        "ambiguous_policy": ambiguous_policy,
        "target_hii_inductive": float(target_comps["inductive"]),
        "target_hii_resonance": float(target_comps["resonance"]),
        "target_hii_aromatic": float(target_comps["aromatic"]),
        "target_hii_steric": float(target_comps["steric"]),
        "target_hii_lonepair": float(target_comps.get("lonepair", 0.0)),
        "target_hii_aromatic_N": float(target_comps.get("aromatic_N", 0.0)),
        "target_hii_field": float(target_comps.get("field", 0.0)),
        "target_hii_aliphatic_total": float(target_comps.get("aliphatic_total", 0.0)),
        "target_hii_total": float(target_comps["total"]),
    })
    row.update(target_meta)

                                                                             
    row.update({
        "site_hii_inductive": row["target_hii_inductive"],
        "site_hii_resonance": row["target_hii_resonance"],
        "site_hii_aromatic": row["target_hii_aromatic"] + row.get("target_hii_aromatic_N", 0.0),
        "site_hii_steric": row["target_hii_steric"] + row.get("target_hii_field", 0.0),
        "site_hii_lonepair": row.get("target_hii_lonepair", 0.0),
        "site_hii_aromatic_N": row.get("target_hii_aromatic_N", 0.0),
        "site_hii_field": row.get("target_hii_field", 0.0),
        "site_hii_aliphatic_total": row.get("target_hii_aliphatic_total", 0.0),
        "site_hii_total": row["target_hii_total"],
        "n_site_acid_pairs": int(row.get("n_acid_pairs", 0)),
        "n_site_base_pairs": int(row.get("n_base_pairs", 0)),
        "acid_type": row.get("acid_center_type", "none"),
        "base_type": row.get("base_center_type", "none"),
        "position": row.get("acid_position", row.get("base_position", "none")),
        "ring_type": row.get("acid_ring_type", row.get("base_ring_type", "none")),
    })

    return row


__all__ = [
    "HALOGENS",
    "HALOGEN_ORDER",
    "HALOGEN_PARAMS",
    "ACID_TOTAL_WEIGHTS",
    "ALIPHATIC_ACID_TOTAL_WEIGHTS",
    "acid_total_v2",
    "acid_aliphatic_total_v1",
    "IonizableCenter",
    "canonicalize_smiles",
    "find_acidic_centers",
    "find_basic_centers",
    "find_ionizable_centers",
    "get_halogen_sites",
    "compute_acid_base_hii_pair_contributions",
    "compute_site_acid_base_hii",
]
