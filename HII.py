                       


from __future__ import annotations

import os
import json
import math
import argparse
import warnings
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
                                                                                       
                                                                                      
FIGURE_EXPORT_DPI = 300
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.dpi"] = FIGURE_EXPORT_DPI
plt.rcParams["savefig.dpi"] = FIGURE_EXPORT_DPI
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
import logging

                                                                           
                                                                             
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"
from sklearn.linear_model import LinearRegression, HuberRegressor, RidgeCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

import torch

from acid_base_hii_utils import (
    HALOGENS,
    HALOGEN_ORDER,
    canonicalize_smiles,
    get_halogen_sites,
    compute_site_acid_base_hii,
)

                                                               
ACID_COMPONENTS = ["inductive", "resonance", "aromatic", "steric"]
ACID_COMPONENT_COLS = [f"acid_hii_{c}_selected" for c in ACID_COMPONENTS]

                                                                             
                                                                            
                                                                       
                                                                               
                                                                 
                                                                             
SCAFFOLD_DESCRIPTOR_COLS = [
    "scaffold_TPSA",
    "scaffold_MolLogP",
    "scaffold_BertzCT",
    "scaffold_HeavyAtomCount",
    "scaffold_HBA",
    "scaffold_HBD",
    "scaffold_RotatableBonds",
    "scaffold_RingCount",
    "scaffold_AromaticRingCount",
    "scaffold_FractionCSP3",
]
                                                                          
                                                                          
SCAFFOLD_COMPONENT_COLS = [
    "scaffold_polarity",
    "scaffold_complexity",
    "scaffold_ring",
]

                                                                              
                                                               
                                                          
SCAFFOLD_TRIPLE_DESCRIPTOR_COLS = list(SCAFFOLD_DESCRIPTOR_COLS)
SCAFFOLD_TRIPLE_SIZE = 3
SCAFFOLD_TRIPLE_MODEL_PREFIX = "acid_fixed_triple_scaffold"

                                                                  
                                                                           
                                                                 
ALIPHATIC_ACID_COMPONENTS: List[str] = []
ALIPHATIC_ACID_COMPONENT_COLS: List[str] = []
ALIPHATIC_ACID_TOTAL_COL = "acid_hii_aliphatic_total_selected"

                                                                 
                                                                  
BASE_COMPONENTS = ["inductive", "lonepair", "aromatic_N", "resonance", "field"]
BASE_COMPONENT_COLS = [f"base_hii_{c}_selected" for c in BASE_COMPONENTS]

                                                                              
                                                                            
                                                                                   
TARGET_COMPONENTS = ["inductive", "resonance", "aromatic", "steric", "lonepair", "aromatic_N", "field"]
TARGET_COMPONENT_COLS = [f"target_hii_{c}" for c in TARGET_COMPONENTS]

                                                                       
                                                                               
HIGH_CONF_ACID_TYPES = {
    "carboxylic_acid",
    "phenol",
    "sulfonamide",
    "sulfonic_acid",
    "thiol",
    "imide",
    "azole_NH",
}

                                                                          
                                                                  
FEATURE_LABELS = {
    "target_hii_total": "HII_target_total",
    "target_hii_inductive": "I_target",
    "target_hii_resonance": "R_target",
    "target_hii_aromatic": "A_target",
    "target_hii_steric": "S_target",
    "target_hii_lonepair": "LP_target",
    "target_hii_aromatic_N": "ArN_target",
    "target_hii_field": "Field_target",
    "acid_hii_inductive_selected": "I_acid",
    "acid_hii_resonance_selected": "R_acid",
    "acid_hii_aromatic_selected": "A_acid",
    "acid_hii_steric_selected": "S_acid",
    "base_hii_inductive_selected": "I_base",
    "base_hii_lonepair_selected": "LP_base",
    "base_hii_aromatic_N_selected": "ArN_base",
    "base_hii_resonance_selected": "R_base",
    "base_hii_field_selected": "Field_base",
    "scaffold_polarity": "S_polar",
    "scaffold_complexity": "S_complexity",
    "scaffold_ring": "S_ring",
    "scaffold_TPSA": "TPSA",
    "scaffold_MolLogP": "MolLogP",
    "scaffold_BertzCT": "BertzCT",
    "scaffold_HeavyAtomCount": "HeavyAtomCount",
    "scaffold_HBA": "HBA",
    "scaffold_HBD": "HBD",
    "scaffold_RotatableBonds": "RotatableBonds",
    "scaffold_RingCount": "RingCount",
    "scaffold_AromaticRingCount": "AromaticRingCount",
    "scaffold_FractionCSP3": "FractionCSP3",
    "mode_acidic": "mode_acidic",
    "mode_basic": "mode_basic",
    "mode_ambiguous": "mode_ambiguous",
}

FINAL_FORMULA_MODEL_PRIORITY = {
                                                                              
    "target_components_plus_pka_mode_interactions": 0,
    "target_hii_components": 1,
    "target_hii_total": 2,
                                                                             
                                                                                 
                                                   
    "acid_fixed_components": 3,
    "acid_directed_components": 4,
    "base_directed_components": 5,
}


                                                                              
 
                                                                     
 
                                                                                 
                                                                              
                                                                          
                                
 
                                                                       
                                                       
 
                                                                            
FINAL_FORMULA_SCALE = 0.64109
FINAL_FORMULA_OFFSET = 0.210521
FINAL_FORMULA_TRANSFORM = "final_formula = fitted_formula / 0.64109 + 0.210521"

PRED_FULL_DATA_CSV = "pred_full_data.csv"
TRUE_FULL_DATA_CSV = "true_full_data.csv"
ACIDIC_PRED_FULL_DATA_CSV = "acidic_pred_full_data_final_formula.csv"
ACIDIC_TRUE_FULL_DATA_CSV = "acidic_true_full_data_final_formula.csv"
FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV = "final_formula_full_data_application_summary.csv"


def apply_final_formula_transform(values: Any) -> np.ndarray:

    arr = np.asarray(values, dtype=float)
    return arr / FINAL_FORMULA_SCALE + FINAL_FORMULA_OFFSET


def final_formula_linear_coefficients(
    intercept: float,
    coef: np.ndarray,
) -> Tuple[float, np.ndarray]:

    final_intercept = float(intercept) / FINAL_FORMULA_SCALE + FINAL_FORMULA_OFFSET
    final_coef = np.asarray(coef, dtype=float).ravel() / FINAL_FORMULA_SCALE
    return final_intercept, final_coef


                                                                                
                                                                               
                                                                   
FEATURE_EXPLANATIONS = {
    "INTERCEPT": "截距项；当所有 HII 描述符和模式指示变量都为 0 时，公式给出的基准 ΔpKa。",
    "target_hii_total": "目标中心 HII 总量；按 pKa 标签模式自动选用 acid-directed 或 base-directed HII 后得到的单一总分。",
    "target_hii_inductive": "目标中心诱导项 I_target；表示卤素通过键/距离传递的吸电子诱导影响。",
    "target_hii_resonance": "目标中心共振项 R_target；表示卤素与目标中心之间可共轭/介观传递的电子影响。",
    "target_hii_aromatic": "目标中心芳香/极化项 A_target；主要用于酸性中心，表示芳香体系与卤素极化性相关的传递项。",
    "target_hii_steric": "目标中心位阻项 S_target；主要用于酸性中心，表示邻近或空间拥挤对 pKa shift 的影响。",
    "target_hii_lonepair": "目标中心孤对电子项 LP_target；主要用于碱性中心，表示卤素对质子化位点孤对电子可用性的影响。",
    "target_hii_aromatic_N": "目标中心芳香氮通信项 ArN_target；主要用于碱性杂芳香 N，表示杂芳香 N 与卤素之间的电子通信。",
    "target_hii_field": "目标中心场效应项 Field_target；主要用于碱性中心，表示短程 through-space/field/conformation 影响。",
    "acid_hii_inductive_selected": "酸性中心诱导项 I_acid；酸性 pKa 位点的 through-bond 吸电子影响。",
    "acid_hii_resonance_selected": "酸性中心共振项 R_acid；酸性中心与卤素之间的共轭/介观传递影响。",
    "acid_hii_aromatic_selected": "酸性中心芳香/极化项 A_acid；芳香体系、环通信和卤素极化性相关影响。",
    "acid_hii_steric_selected": "酸性中心位阻项 S_acid；邻位/近距离位阻对酸性中心 ΔpKa 的影响。",
    "base_hii_inductive_selected": "碱性中心诱导项 I_base；卤素对碱性中心电子密度的 through-bond 吸电子影响。",
    "base_hii_lonepair_selected": "碱性中心孤对电子项 LP_base；碱性中心质子化孤对电子可及性/可用性的影响。",
    "base_hii_aromatic_N_selected": "碱性中心芳香氮通信项 ArN_base；杂芳香 N 与卤素之间的电子通信影响。",
    "base_hii_resonance_selected": "碱性中心共振项 R_base；碱性中心的介观/共轭路径项，其方向由拟合系数决定。",
    "base_hii_field_selected": "碱性中心场效应项 Field_base；短程 through-space/field/conformational 影响。",
    "scaffold_polarity": "Scaffold 极性/氢键背景项 S_polar；由 masked parent R-H 的 TPSA、HBA、HBD 和 MolLogP 组合得到，表示除 focus 卤素后的整体极性与氢键能力背景。",
    "scaffold_complexity": "Scaffold 复杂度项 S_complexity；由 masked parent R-H 的 BertzCT、重原子数、旋转键数和 FractionCSP3 组合得到，表示骨架大小、拓扑和构象复杂度。",
    "scaffold_ring": "Scaffold 环/芳香背景项 S_ring；由 masked parent R-H 的 RingCount 和 AromaticRingCount 组合得到，表示环和芳香骨架背景。",
    "scaffold_TPSA": "Masked parent R-H 的 TPSA；极性表面积，表示母体去卤素后整体极性表面背景。",
    "scaffold_MolLogP": "Masked parent R-H 的 MolLogP；疏水/亲水平衡，数值越大通常越疏水。",
    "scaffold_BertzCT": "Masked parent R-H 的 BertzCT；拓扑复杂度，表示骨架连接复杂性。",
    "scaffold_HeavyAtomCount": "Masked parent R-H 的 HeavyAtomCount；母体去卤素后的重原子数，即 scaffold 大小。",
    "scaffold_HBA": "Masked parent R-H 的氢键受体数量；RDKit CalcNumHBA。",
    "scaffold_HBD": "Masked parent R-H 的氢键供体数量；RDKit CalcNumHBD。",
    "scaffold_RotatableBonds": "Masked parent R-H 的可旋转键数量；表示构象自由度。",
    "scaffold_RingCount": "Masked parent R-H 的总环数量；表示环骨架背景。",
    "scaffold_AromaticRingCount": "Masked parent R-H 的芳香环数量；表示芳香 scaffold 背景。",
    "scaffold_FractionCSP3": "Masked parent R-H 的 FractionCSP3；表示 sp3/脂肪性/饱和程度。",
    "mode_acidic": "pKa 模式指示变量；该行被推断为 acidic 时取 1，否则取 0。",
    "mode_basic": "pKa 模式指示变量；该行被推断为 basic 时取 1，否则取 0。",
    "mode_ambiguous": "pKa 模式指示变量；该行被推断为 ambiguous 时取 1，否则取 0。",
}

MODEL_FORMULA_EXPLANATIONS = {
    "target_hii_total": "最终公式只使用目标中心 HII 总量，形式为 ΔpKa = b0 + b_total·HII_target_total。",
    "target_hii_components": "最终公式使用目标中心 HII 分量；acidic/basic 行会按 target-center 选择对应分量。",
    "acid_directed_components": "最终公式使用 acid-directed 四个分量 I/R/A/S；适合解释酸性中心子集。",
    "base_directed_components": "最终公式使用 base-directed 五个分量 I/LP/ArN/R/Field；适合解释碱性中心子集。",
    "acid_fixed_components": "旧版 scaffold 合成项公式：ΔpKa = b0 + bI·I + bR·R + bA·A + bS·S + βP·S_polar + βC·S_complexity + βR·S_ring。",
    SCAFFOLD_TRIPLE_MODEL_PREFIX: "三三组合 scaffold 公式：ΔpKa = b0 + bI·I + bR·R + bA·A + bS·S + β1·D1 + β2·D2 + β3·D3；D1/D2/D3 从 10 个 masked-parent RDKit descriptor 中选择。",
    "target_components_plus_pka_mode_interactions": "最终公式使用目标中心 HII 分量、pKa 模式指示变量，以及 HII × pKa 模式交互项，用来描述 acidic/basic/ambiguous 的不同传递关系。",
}


def fmt_float(v: Any, precision: int = 6) -> str:

    try:
        x = float(v)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{int(precision)}g}"


def _component_short_to_label(short: str) -> str:
    return {
        "I": "I_acid",
        "R": "R_acid",
        "A": "A_acid",
        "S": "S_acid",
    }.get(str(short), str(short))


def feature_label(col: str) -> str:

    col = str(col)
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]

                                                                                     
                                                             
    if col.startswith("acidctx__"):
        body = col.replace("acidctx__", "", 1)
        if "__" in body:
            context_token, comp_short = body.rsplit("__", 1)
            halogen = None
            acid_type = None
            for h in HALOGEN_ORDER:
                prefix = f"{h}_"
                if context_token == h:
                    halogen = h
                    acid_type = "unknown_acid"
                    break
                if context_token.startswith(prefix):
                    halogen = h
                    acid_type = context_token[len(prefix):]
                    break
            if halogen is None:
                parts = context_token.split("_", 1)
                halogen = parts[0]
                acid_type = parts[1] if len(parts) > 1 else "unknown_acid"
            return f"{_component_short_to_label(comp_short)}[{halogen},{acid_type}]"

                                                                              
    if "_x_" in col:
        base, mode = col.rsplit("_x_", 1)
        return f"{feature_label(base)}*mode_{mode}"

    return col


def format_linear_formula(
    target: str,
    intercept: float,
    coef_by_col: Dict[str, float],
    precision: int = 6,
) -> str:

    try:
        b0 = float(intercept)
    except Exception:
        return ""
    if not np.isfinite(b0):
        return ""
    formula = f"{target} = {fmt_float(b0, precision)}"
    for col, coef in coef_by_col.items():
        try:
            c = float(coef)
        except Exception:
            continue
        if not np.isfinite(c):
            continue
        sign = "+" if c >= 0 else "-"
        formula += f" {sign} {fmt_float(abs(c), precision)}*{feature_label(col)}"
    return formula


def add_formula_to_record(
    rec: Dict[str, Any],
    target: str,
    x_cols: List[str],
    raw_intercept: float,
    raw_coef: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:


    coef_before = np.asarray(raw_coef, dtype=float).ravel()
    final_intercept, final_coef = final_formula_linear_coefficients(raw_intercept, coef_before)

    rec["final_formula_transform"] = FINAL_FORMULA_TRANSFORM
    rec["final_formula_scale_divisor"] = float(FINAL_FORMULA_SCALE)
    rec["final_formula_offset"] = float(FINAL_FORMULA_OFFSET)
    rec["intercept_before_final_transform"] = float(raw_intercept)
    rec["formula_before_final_transform"] = format_linear_formula(
        target=target,
        intercept=float(raw_intercept),
        coef_by_col={col: float(val) for col, val in zip(x_cols, coef_before)},
        precision=int(getattr(args, "formula_precision", 6)),
    )

    coef_by_col: Dict[str, float] = {}
    for col, before_val, final_val in zip(x_cols, coef_before, final_coef):
        rec[f"coef_before_final_transform__{col}"] = float(before_val)
        coef_by_col[col] = float(final_val)
        rec[f"coef_raw__{col}"] = float(final_val)

    rec["raw_intercept"] = float(final_intercept)
    rec["formula"] = format_linear_formula(
        target=target,
        intercept=float(final_intercept),
        coef_by_col=coef_by_col,
        precision=int(getattr(args, "formula_precision", 6)),
    )
    return rec


def explain_feature(col: str) -> str:

    col = str(col)
    if col in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[col]

    if col.startswith("acidctx__"):
        body = col.replace("acidctx__", "", 1)
        if "__" in body:
            context_token, comp_short = body.rsplit("__", 1)
            halogen = None
            acid_type = None
            for h in HALOGEN_ORDER:
                prefix = f"{h}_"
                if context_token == h:
                    halogen = h
                    acid_type = "unknown_acid"
                    break
                if context_token.startswith(prefix):
                    halogen = h
                    acid_type = context_token[len(prefix):]
                    break
            if halogen is None:
                parts = context_token.split("_", 1)
                halogen = parts[0]
                acid_type = parts[1] if len(parts) > 1 else "unknown_acid"
            comp_label = _component_short_to_label(comp_short)
            return (
                f"酸性上下文交互参数；仅当 halogen={halogen} 且 acid_center_type={acid_type} 时，"
                f"该行的 {comp_label} 分量进入公式，否则该项为 0。"
            )

    if "_x_" in col:
        base, mode = col.rsplit("_x_", 1)
        return (
            f"pKa 模式交互项；仅当 pka_mode_inferred={mode} 时启用，"
            f"数值等于 {feature_label(base)}，否则为 0。基础含义：{explain_feature(base)}"
        )

    return "拟合公式中的数值描述符；系数表示该描述符每增加 1 个原始单位时 ΔpKa 的线性变化。"


def parameter_activation_rule(col: str) -> str:

    col = str(col)
    if col == "INTERCEPT":
        return "always"
    if col.startswith("mode_"):
        return f"1 if pka_mode_inferred == {col.replace('mode_', '')}, else 0"
    if col.startswith("acidctx__"):
        label = feature_label(col)
        inside = label[label.find("[") + 1:label.find("]")] if "[" in label and "]" in label else "context"
        return f"active only for acid context [{inside}]"
    if "_x_" in col:
        _, mode = col.rsplit("_x_", 1)
        return f"active only if pka_mode_inferred == {mode}"
    return "uses the raw feature value in each counterfactual row"


def formula_term_text(raw_feature: str, coefficient: float, precision: int = 6) -> str:

    if raw_feature == "INTERCEPT":
        return fmt_float(coefficient, precision)
    sign = "+" if float(coefficient) >= 0 else "-"
    return f"{sign} {fmt_float(abs(float(coefficient)), precision)}*{feature_label(raw_feature)}"


def _find_matching_fit_row(fit_df: pd.DataFrame, selected_row: pd.Series) -> Optional[pd.Series]:

    if fit_df is None or fit_df.empty:
        return None
    mask = pd.Series(True, index=fit_df.index)
    for key in ["target", "subset", "model"]:
        if key in fit_df.columns and key in selected_row.index:
            mask &= fit_df[key].astype(str) == str(selected_row.get(key))
    candidates = fit_df[mask].copy()
    if candidates.empty:
        return None
    if "full_data_r2" in candidates.columns and "full_data_r2" in selected_row.index:
        selected_r2 = selected_row.get("full_data_r2", np.nan)
        try:
            selected_r2 = float(selected_r2)
            candidates["_r2_abs_diff"] = (pd.to_numeric(candidates["full_data_r2"], errors="coerce") - selected_r2).abs()
            candidates = candidates.sort_values("_r2_abs_diff", ascending=True)
        except Exception:
            pass
    return candidates.iloc[0]


def build_formula_parameter_table(
    selected_formula_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    args: argparse.Namespace,
    table_scope: str,
) -> pd.DataFrame:

    columns = [
        "table_scope", "target", "subset", "model", "n", "n_features",
        "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule",
        "term_order", "parameter_kind", "raw_feature", "display_name",
        "coefficient", "abs_coefficient", "sign", "formula_term",
        "activation_rule", "explanation", "model_explanation", "formula",
    ]
    if selected_formula_df is None or selected_formula_df.empty:
        return pd.DataFrame(columns=columns)

    precision = int(getattr(args, "formula_precision", 6))
    rows: List[Dict[str, Any]] = []
    for _, selected in selected_formula_df.iterrows():
        fit_row = _find_matching_fit_row(fit_df, selected)
        if fit_row is None:
            continue

        meta = {
            "table_scope": table_scope,
            "target": selected.get("target", fit_row.get("target", "")),
            "subset": selected.get("subset", fit_row.get("subset", "")),
            "model": selected.get("model", fit_row.get("model", "")),
            "n": selected.get("n", fit_row.get("n", np.nan)),
            "n_features": selected.get("n_features", fit_row.get("n_features", np.nan)),
            "full_data_r2": selected.get("full_data_r2", fit_row.get("full_data_r2", np.nan)),
            "full_data_rmse": selected.get("full_data_rmse", fit_row.get("full_data_rmse", np.nan)),
            "full_data_mae": selected.get("full_data_mae", fit_row.get("full_data_mae", np.nan)),
            "selection_rule": selected.get("selection_rule", ""),
            "model_explanation": model_formula_explanation(selected.get("model", fit_row.get("model", ""))),
            "formula": selected.get("formula", fit_row.get("formula", "")),
        }

        intercept = fit_row.get("raw_intercept", np.nan)
        if pd.notna(intercept):
            coef = float(intercept)
            rows.append({
                **meta,
                "term_order": 0,
                "parameter_kind": "intercept",
                "raw_feature": "INTERCEPT",
                "display_name": "Intercept",
                "coefficient": coef,
                "abs_coefficient": abs(coef),
                "sign": "+" if coef >= 0 else "-",
                "formula_term": formula_term_text("INTERCEPT", coef, precision),
                "activation_rule": parameter_activation_rule("INTERCEPT"),
                "explanation": explain_feature("INTERCEPT"),
            })

        x_cols_raw = str(fit_row.get("x_cols", ""))
        x_cols = [c for c in x_cols_raw.split("|") if c and c != "nan"]
        coef_cols = [c for c in fit_row.index if str(c).startswith("coef_raw__")]
        coef_feature_order = [c.replace("coef_raw__", "", 1) for c in coef_cols]
        if not x_cols:
            x_cols = coef_feature_order

                                                                                  
                                                                            
        ordered_features: List[str] = []
        for f in x_cols + coef_feature_order:
            if f not in ordered_features:
                ordered_features.append(f)

        for idx, raw_feature in enumerate(ordered_features, start=1):
            coef_col = f"coef_raw__{raw_feature}"
            if coef_col not in fit_row.index:
                continue
            val = fit_row.get(coef_col, np.nan)
            if pd.isna(val):
                continue
            coef = float(val)
            rows.append({
                **meta,
                "term_order": idx,
                "parameter_kind": "coefficient",
                "raw_feature": raw_feature,
                "display_name": feature_label(raw_feature),
                "coefficient": coef,
                "abs_coefficient": abs(coef),
                "sign": "+" if coef >= 0 else "-",
                "formula_term": formula_term_text(raw_feature, coef, precision),
                "activation_rule": parameter_activation_rule(raw_feature),
                "explanation": explain_feature(raw_feature),
            })

    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["target", "subset", "model", "term_order"]).reset_index(drop=True)
    return out


def build_formula_parameter_explanation_table() -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []
    for raw_feature, label in FEATURE_LABELS.items():
        rows.append({
            "raw_feature": raw_feature,
            "display_name": label,
            "activation_rule": parameter_activation_rule(raw_feature),
            "explanation": explain_feature(raw_feature),
        })
    rows.append({
        "raw_feature": "INTERCEPT",
        "display_name": "Intercept",
        "activation_rule": parameter_activation_rule("INTERCEPT"),
        "explanation": explain_feature("INTERCEPT"),
    })
    return pd.DataFrame(rows).sort_values(["raw_feature"]).reset_index(drop=True)


def write_final_formula_parameter_report(
    final_formula_df: pd.DataFrame,
    final_parameter_df: pd.DataFrame,
    final_by_subset_parameter_df: pd.DataFrame,
    explanation_df: pd.DataFrame,
    output_dir: str,
) -> None:

    lines: List[str] = []
    lines.append("# Final fitted formula and parameter values")
    lines.append("")
    lines.append("This report is generated after fitting the final formula on the complete counterfactual full-data table produced by the script.")
    lines.append("The coefficients below are in raw feature units, not standardized units, so they can be read directly in the displayed formula.")
    lines.append("")

    if final_formula_df is not None and not final_formula_df.empty:
        lines.append("## Main final formula")
        lines.append("")
        show = [c for c in ["target", "subset", "model", "n", "n_features", "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule", "formula"] if c in final_formula_df.columns]
        lines.append(final_formula_df[show].to_markdown(index=False))
        lines.append("")

    if final_parameter_df is not None and not final_parameter_df.empty:
        lines.append("## Main final formula parameters")
        lines.append("")
        for (target, subset, model), sub in final_parameter_df.groupby(["target", "subset", "model"], dropna=False):
            lines.append(f"### target={target}, subset={subset}, model={model}")
            lines.append("")
            show = ["term_order", "parameter_kind", "display_name", "coefficient", "activation_rule", "explanation"]
            lines.append(sub[show].to_markdown(index=False))
            lines.append("")

    if final_by_subset_parameter_df is not None and not final_by_subset_parameter_df.empty:
        lines.append("## Subset final formula parameters")
        lines.append("")
        for (target, subset, model), sub in final_by_subset_parameter_df.groupby(["target", "subset", "model"], dropna=False):
            lines.append(f"### target={target}, subset={subset}, model={model}")
            lines.append("")
            show = ["term_order", "parameter_kind", "display_name", "coefficient", "activation_rule", "explanation"]
            lines.append(sub[show].to_markdown(index=False))
            lines.append("")

    if explanation_df is not None and not explanation_df.empty:
        lines.append("## Symbol glossary")
        lines.append("")
        lines.append(explanation_df[["raw_feature", "display_name", "activation_rule", "explanation"]].to_markdown(index=False))
        lines.append("")

    with open(os.path.join(output_dir, "final_formula_with_parameters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_final_formula_parameter_details(
    fit_df: pd.DataFrame,
    final_formula_df: pd.DataFrame,
    final_formula_by_subset_df: pd.DataFrame,
    output_dir: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    final_parameter_df = build_formula_parameter_table(
        selected_formula_df=final_formula_df,
        fit_df=fit_df,
        args=args,
        table_scope="main_final_formula",
    )
    final_by_subset_parameter_df = build_formula_parameter_table(
        selected_formula_df=final_formula_by_subset_df,
        fit_df=fit_df,
        args=args,
        table_scope="subset_final_formula",
    )
    explanation_df = build_formula_parameter_explanation_table()

    final_parameter_df.to_csv(os.path.join(output_dir, "final_formula_parameters.csv"), index=False)
    final_by_subset_parameter_df.to_csv(os.path.join(output_dir, "final_formula_by_subset_parameters.csv"), index=False)
    explanation_df.to_csv(os.path.join(output_dir, "formula_parameter_explanations.csv"), index=False)
    write_final_formula_parameter_report(
        final_formula_df=final_formula_df,
        final_parameter_df=final_parameter_df,
        final_by_subset_parameter_df=final_by_subset_parameter_df,
        explanation_df=explanation_df,
        output_dir=output_dir,
    )
    return final_parameter_df, final_by_subset_parameter_df, explanation_df


def export_final_hii_formula_outputs(
    final_formula_df: pd.DataFrame,
    final_formula_by_subset_df: pd.DataFrame,
    final_parameter_df: pd.DataFrame,
    final_by_subset_parameter_df: pd.DataFrame,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:


    formula_frames: List[pd.DataFrame] = []
    if final_formula_df is not None and not final_formula_df.empty:
        f = final_formula_df.copy()
        f.insert(0, "formula_scope", "main_final_formula")
        formula_frames.append(f)
    if final_formula_by_subset_df is not None and not final_formula_by_subset_df.empty:
        f = final_formula_by_subset_df.copy()
        f.insert(0, "formula_scope", "subset_final_formula")
        formula_frames.append(f)
    formula_out = pd.concat(formula_frames, ignore_index=True) if formula_frames else pd.DataFrame()
    formula_out.to_csv(os.path.join(output_dir, "final_hii_formula_summary.csv"), index=False)

    parameter_frames: List[pd.DataFrame] = []
    if final_parameter_df is not None and not final_parameter_df.empty:
        parameter_frames.append(final_parameter_df.copy())
    if final_by_subset_parameter_df is not None and not final_by_subset_parameter_df.empty:
        parameter_frames.append(final_by_subset_parameter_df.copy())
    parameter_out = pd.concat(parameter_frames, ignore_index=True) if parameter_frames else pd.DataFrame()
    parameter_out.to_csv(os.path.join(output_dir, "final_hii_formula_parameters.csv"), index=False)

    lines: List[str] = []
    lines.append("# Final HII fitted formula, parameter values, and meanings")
    lines.append("")
    lines.append("This file is an explicit HII-formula view of the final fitted formulas. Coefficients are in raw HII feature units, not standardized units.")
    lines.append("")
    if not formula_out.empty:
        show = [c for c in [
            "formula_scope", "target", "subset", "model", "n", "n_features",
            "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule", "formula",
        ] if c in formula_out.columns]
        lines.append("## Formula summary")
        lines.append("")
        lines.append(formula_out[show].to_markdown(index=False))
        lines.append("")
    if not parameter_out.empty:
        lines.append("## Parameter table")
        lines.append("")
        group_cols = [c for c in ["table_scope", "target", "subset", "model"] if c in parameter_out.columns]
        if group_cols:
            for keys, sub in parameter_out.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                title = ", ".join(f"{c}={v}" for c, v in zip(group_cols, keys))
                lines.append(f"### {title}")
                lines.append("")
                show = [c for c in ["term_order", "parameter_kind", "display_name", "raw_feature", "coefficient", "activation_rule", "explanation"] if c in sub.columns]
                lines.append(sub[show].to_markdown(index=False))
                lines.append("")
        else:
            lines.append(parameter_out.to_markdown(index=False))
    with open(os.path.join(output_dir, "final_hii_formula_with_parameters.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return formula_out, parameter_out


def write_selected_formula_parameter_report(
    selected_formula_df: pd.DataFrame,
    parameter_df: pd.DataFrame,
    output_path: str,
    title: str,
) -> None:

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("Coefficients are in raw HII feature units. The table below gives every parameter value and its activation/meaning.")
    lines.append("")
    if selected_formula_df is not None and not selected_formula_df.empty:
        show = [c for c in [
            "target", "applied_to_target", "formula_source_target", "subset", "model", "n", "n_features",
            "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule", "formula",
        ] if c in selected_formula_df.columns]
        if show:
            lines.append("## Selected formula")
            lines.append("")
            lines.append(selected_formula_df[show].to_markdown(index=False))
            lines.append("")
    if parameter_df is not None and not parameter_df.empty:
        lines.append("## Parameters")
        lines.append("")
        show = [c for c in ["term_order", "parameter_kind", "display_name", "raw_feature", "coefficient", "activation_rule", "explanation"] if c in parameter_df.columns]
        lines.append(parameter_df[show].to_markdown(index=False))
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_selected_formula_parameter_details(
    fit_df: pd.DataFrame,
    selected_row: pd.Series,
    output_dir: str,
    args: argparse.Namespace,
    file_prefix: str,
    table_scope: str,
) -> pd.DataFrame:

    selected_df = pd.DataFrame([selected_row.drop(labels=["prio"], errors="ignore")])
    if "selection_rule" not in selected_df.columns:
        selected_df["selection_rule"] = table_scope
    summary_path = os.path.join(output_dir, f"{file_prefix}_formula_summary.csv")
    selected_df.to_csv(summary_path, index=False)

    parameter_df = build_formula_parameter_table(
        selected_formula_df=selected_df,
        fit_df=fit_df,
        args=args,
        table_scope=table_scope,
    )
    parameter_path = os.path.join(output_dir, f"{file_prefix}_formula_parameters.csv")
    parameter_df.to_csv(parameter_path, index=False)
    write_selected_formula_parameter_report(
        selected_formula_df=selected_df,
        parameter_df=parameter_df,
        output_path=os.path.join(output_dir, f"{file_prefix}_formula_with_parameters.md"),
        title=f"{file_prefix}: selected HII formula and parameters",
    )
    return parameter_df


                                                                               
     
                                                                               

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acid/base-aware counterfactual halogen-mask MMP-HII analysis for a trained pKa GNN."
    )

                            
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_ckpt.pth.")
    parser.add_argument("--cfg", type=str, default=None, help="Fallback YAML config path if checkpoint has no embedded config.")
    parser.add_argument("--no-checkpoint-config", action="store_true", help="Force YAML config instead of checkpoint['config'].")
    parser.add_argument("--use-optimized-config", action="store_true", help="Use optimized_config.py for YAML fallback.")
    parser.add_argument("--opts", nargs="+", default=None, help="Optional KEY VALUE overrides, e.g. DATA.DATA_PATH ./data DATA.DATASET pka.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size override.")
    parser.add_argument("--strict-load", action="store_true", help="Use strict model weight loading.")

                   
    parser.add_argument("--input-csv", type=str, default=None, help="Optional input CSV. Defaults to cfg.DATA.DATA_PATH/raw/{cfg.DATA.DATASET}.csv.")
    parser.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column in input CSV.")
    parser.add_argument("--target-task", type=str, default=None, help="pKa task column/name. Defaults to first task in cfg/raw CSV.")
    parser.add_argument("--pka-type-col", type=str, default=None, help="Optional column with acidic/basic label for each pKa row.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")

                                      
    parser.add_argument("--max-sites", type=int, default=None, help="Optional cap on number of halogen sites for debugging.")
    parser.add_argument("--only-aromatic-halogen", action="store_true", help="Analyze only halogens attached to aromatic atoms.")
    parser.add_argument("--hii-aggregation", choices=["target_weighted", "pair_mean", "sum"], default="target_weighted", help="Primary HII aggregation for target-center HII. target_weighted uses center priors, distance and pKa plausibility.")
    parser.add_argument("--hii-weight-tau", type=float, default=0.75, help="Softmax temperature for target-center weighting; lower values select a more dominant center.")
    parser.add_argument(
        "--ambiguous-policy",
        choices=["mixed", "max_hii", "acid", "base"],
        default="mixed",
        help="How to select target HII for molecules with both acid and base centers and no explicit pKa type.",
    )
    parser.add_argument(
        "--unknown-policy",
        choices=["zero"],
        default="zero",
        help=(
            "Compatibility option for v3 commands. In this v2-clean script, unknown pKa mode "
            "always keeps target HII at zero; no halogen fallback is applied."
        ),
    )
    parser.add_argument("--min-group-size", type=int, default=5, help="Minimum group size for summaries/fits.")
    parser.add_argument(
        "--disable-acid-fixed",
        "--disable-acid-context",
        dest="disable_acid_fixed",
        action="store_true",
        help="Disable the acid_fixed_components model. The --disable-acid-context alias is kept for older commands.",
    )
    parser.add_argument(
        "--acid-fixed-min-n",
        "--acid-context-min-n",
        dest="acid_fixed_min_n",
        type=int,
        default=5,
        help=(
            "Minimum rows required before fitting the acid fixed-coefficient model "
            "within a subset. This is independent of --min-group-size because the fixed-coefficient model has several terms."
        ),
    )
    parser.add_argument(
        "--acid-fixed-alpha",
        "--acid-context-alpha",
        dest="acid_fixed_alpha",
        type=float,
        default=30.0,
        help=(
            "Ridge alpha for acid_fixed_components. Default 30.0 is a conservative fixed regularization "
            "chosen to stabilize the global fixed acid coefficient model under grouped CV."
        ),
    )

                       
    parser.add_argument("--fit-model", choices=["linear", "huber", "ridge"], default="linear", help="Interpretable regression model for HII fits.")
    parser.add_argument("--cv", choices=["group", "kfold", "none"], default="group", help="Cross-validation mode for linear HII models.")
    parser.add_argument("--group-col", default="masked_parent_smiles", help="Group column for GroupKFold.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--main-formula-model",
        choices=[
            "auto_best_full_data",
            "target_hii_total",
            "target_hii_components",
            "acid_directed_components",
            "base_directed_components",
            "acid_fixed_components",
            "target_components_plus_pka_mode_interactions",
        ],
        default="auto_best_full_data",
        help=(
            "Which fitted formula is treated as the primary final formula. "
            "auto_best_full_data selects the highest full-data R² among global all-data formulas."
        ),
    )
    parser.add_argument("--formula-precision", type=int, default=6, help="Significant digits used in exported formula strings.")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG figures.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help=(
            "Figure DPI option retained for command compatibility. "
            "PNG export is forced to 300 dpi and each PNG also gets a same-name SVG."
        ),
    )

    return parser.parse_args()


                                                                               
                               
                                                                               

class _ConfigArgs:
    def __init__(self, cfg: str, opts=None, batch_size=None):
        self.cfg = cfg
        self.opts = opts
        self.batch_size = batch_size
        self.lr_scheduler = None
        self.resume = None
        self.tag = None
        self.eval = False


def _import_project_modules(use_optimized_config: bool = False):
    if use_optimized_config:
        from optimized_config import get_config
    else:
        from config import get_config
    from utils import get_task_names
    from model import build_model
    return get_config, get_task_names, build_model


def _torch_load_checkpoint(checkpoint_path: str):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "yacs":
            raise ModuleNotFoundError(
                "Loading checkpoint['config'] requires yacs. Install with `pip install yacs`, "
                "or run in the original training environment."
            ) from exc
        raise


def _cfg_get(cfg, dotted_key: str, default=None):
    cur = cfg
    for p in dotted_key.split("."):
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur


def _apply_overrides(cfg, args: argparse.Namespace) -> None:
    if not hasattr(cfg, "defrost"):
        return
    cfg.defrost()
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    cfg.freeze()


def _state_dict_from_ckpt(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def _infer_and_apply_architecture_from_checkpoint(cfg, ckpt) -> List[str]:
    changes: List[str] = []
    if ckpt is None or not hasattr(cfg, "defrost"):
        return changes
    state = _state_dict_from_ckpt(ckpt)
    if not isinstance(state, dict):
        return changes

    cfg.defrost()
    lin_a = state.get("lin_a.weight", None)
    if lin_a is not None and hasattr(lin_a, "shape") and len(lin_a.shape) == 2:
        hidden = int(lin_a.shape[0])
        if hasattr(cfg.MODEL, "HID") and int(cfg.MODEL.HID) != hidden:
            old = int(cfg.MODEL.HID)
            cfg.MODEL.HID = hidden
            changes.append(f"MODEL.HID {old} -> {hidden} inferred from lin_a.weight")

    hidden = int(cfg.MODEL.HID) if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "HID") else None
    out0 = state.get("out.0.weight", None)
    if out0 is not None and hasattr(out0, "shape") and len(out0.shape) == 2 and hidden is not None:
        in_dim = int(out0.shape[1])
        if in_dim == 2 * hidden:
            if hasattr(cfg.MODEL, "THREE_LEVEL") and bool(cfg.MODEL.THREE_LEVEL.ENABLE):
                cfg.MODEL.THREE_LEVEL.ENABLE = False
                changes.append("MODEL.THREE_LEVEL.ENABLE True -> False inferred from out.0.weight=[H,2H]")
            if hasattr(cfg.MODEL, "BRICS") and not bool(cfg.MODEL.BRICS):
                cfg.MODEL.BRICS = True
                changes.append("MODEL.BRICS False -> True inferred from two-level BRICS head")
        elif in_dim == hidden:
            if hasattr(cfg.MODEL, "THREE_LEVEL") and not bool(cfg.MODEL.THREE_LEVEL.ENABLE):
                cfg.MODEL.THREE_LEVEL.ENABLE = True
                changes.append("MODEL.THREE_LEVEL.ENABLE False -> True inferred from out.0.weight=[H,H]")
        else:
            changes.append(f"WARNING cannot infer output head from out.0.weight shape={tuple(out0.shape)} H={hidden}")
    cfg.freeze()
    return changes


def _resolve_task_names_and_outdim(cfg, get_task_names) -> List[str]:
    task_names = _cfg_get(cfg, "DATA.TASK_NAME", None)
    if task_names is None or len(task_names) == 0:
        raw_csv = os.path.join(cfg.DATA.DATA_PATH, "raw", f"{cfg.DATA.DATASET}.csv")
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Cannot infer task names. Raw CSV not found: {raw_csv}")
        task_names = get_task_names(raw_csv)
    else:
        task_names = list(task_names)

    if hasattr(cfg, "defrost"):
        cfg.defrost()
        cfg.DATA.TASK_NAME = list(task_names)
        cfg.MODEL.OUT_DIM = 2 * len(task_names) if cfg.DATA.TASK_TYPE == "classification" else len(task_names)
        cfg.freeze()
    return list(task_names)


def _load_cfg_and_model(args: argparse.Namespace):
    get_config, get_task_names, build_model = _import_project_modules(args.use_optimized_config)
    ckpt = _torch_load_checkpoint(args.checkpoint)

    if isinstance(ckpt, dict) and (not args.no_checkpoint_config) and "config" in ckpt and ckpt["config"] is not None:
        cfg = ckpt["config"]
        cfg_source = "checkpoint['config']"
        _apply_overrides(cfg, args)
    else:
        if not args.cfg:
            raise ValueError("Checkpoint has no embedded config. Provide --cfg pka.yaml.")
        cfg = get_config(_ConfigArgs(args.cfg, opts=args.opts, batch_size=args.batch_size))
        cfg_source = f"yaml:{args.cfg}"

    arch_changes = _infer_and_apply_architecture_from_checkpoint(cfg, ckpt)
    task_names = _resolve_task_names_and_outdim(cfg, get_task_names)

    model = build_model(cfg)
    state = _state_dict_from_ckpt(ckpt)
    try:
        load_msg = model.load_state_dict(state, strict=args.strict_load)
    except RuntimeError as exc:
        if args.strict_load:
            raise
        warnings.warn(f"Strict checkpoint loading failed; retrying strict=False. Error: {exc}")
        load_msg = model.load_state_dict(state, strict=False)

    return cfg, task_names, model, ckpt, cfg_source, arch_changes, load_msg


                                                                               
                                                
                                                                               

def mask_halogen_to_parent_smiles(smiles: str, halogen_idx: int) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or halogen_idx < 0 or halogen_idx >= mol.GetNumAtoms():
        return None
    atom = mol.GetAtomWithIdx(int(halogen_idx))
    if atom.GetSymbol() not in HALOGENS:
        return None
    for sanitize_mode in ["strict", "catch"]:
        try:
            rw = Chem.RWMol(mol)
            rw.RemoveAtom(int(halogen_idx))
            parent = rw.GetMol()
            if sanitize_mode == "strict":
                Chem.SanitizeMol(parent)
            else:
                Chem.SanitizeMol(parent, catchErrors=True)
            return Chem.MolToSmiles(parent, isomericSmiles=True)
        except Exception:
            continue
    return None


def mol_to_graph_data(smiles: str, cfg):
    from dataset import (
        EnhancedMolData,
        atom_attr,
        bond_attr,
        bond_break,
        calculate_molecular_descriptors,
        normalize_descriptors,
        extract_pharmacophores,
        build_pharmacophore_graph,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    use_pharmacophores = bool(hasattr(cfg.MODEL, "THREE_LEVEL") and cfg.MODEL.THREE_LEVEL.ENABLE)
    if use_pharmacophores:
        try:
            mol_3d = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol_3d, randomSeed=42, maxAttempts=50)
            if result == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
                except Exception:
                    pass
        except Exception:
            pass

    try:
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = bond_break(mol)
    except Exception as exc:
        warnings.warn(f"Graph conversion failed for {smiles}: {exc}")
        return None

    descriptor_types = []
    if hasattr(cfg, "DESCRIPTOR") and cfg.DESCRIPTOR.ENABLE:
        descriptor_types = list(cfg.DESCRIPTOR.TYPES)
    descriptors = None
    if descriptor_types:
        desc_raw = calculate_molecular_descriptors(mol, descriptor_types)
        desc = normalize_descriptors(desc_raw, descriptor_types)
        descriptors = torch.FloatTensor(desc)

    pharm_data: Dict[str, Any] = {}
    if use_pharmacophores:
        pharmacophores, atom_to_pharm_dict = extract_pharmacophores(mol)
        if len(pharmacophores) > 0:
            pharm_x = torch.FloatTensor([p["type_encoding"] + [p["size"] / 10.0] for p in pharmacophores])
            pharm_edges, pharm_edge_attrs = build_pharmacophore_graph(pharmacophores)
            if len(pharm_edges) > 0:
                pharm_edge_index = torch.LongTensor(pharm_edges).t().contiguous()
                pharm_edge_attr = torch.FloatTensor(pharm_edge_attrs)
            else:
                pharm_edge_index = torch.zeros((2, 0), dtype=torch.long)
                pharm_edge_attr = torch.zeros((0, 3), dtype=torch.float)

            atom_to_pharm_edges = []
            for atom_idx, pharm_ids in atom_to_pharm_dict.items():
                for pid in pharm_ids:
                    atom_to_pharm_edges.append([int(atom_idx), int(pid)])
            atom_to_pharm = torch.LongTensor(atom_to_pharm_edges) if atom_to_pharm_edges else torch.zeros((0, 2), dtype=torch.long)
            pharm_batch = torch.zeros(len(pharmacophores), dtype=torch.long)
            pharm_data = {
                "pharm_x": pharm_x,
                "pharm_edge_index": pharm_edge_index,
                "pharm_edge_attr": pharm_edge_attr,
                "pharm_batch": pharm_batch,
                "atom_to_pharm": atom_to_pharm,
            }

    if not isinstance(cluster_index, torch.Tensor):
        cluster_index = torch.LongTensor([]) if cluster_index == [] or cluster_index is None else torch.LongTensor(cluster_index)

    data = EnhancedMolData(
        x=torch.FloatTensor(node_attr),
        edge_index=torch.LongTensor(edge_index).t().contiguous(),
        edge_attr=torch.FloatTensor(edge_attr),
        fra_edge_index=torch.LongTensor(fra_edge_index).t().contiguous(),
        fra_edge_attr=torch.FloatTensor(fra_edge_attr),
        cluster_index=cluster_index,
        descriptors=descriptors,
        y=torch.zeros((1, int(cfg.MODEL.OUT_DIM)), dtype=torch.float),
        smiles=canonicalize_smiles(smiles) or smiles,
        **pharm_data,
    )
    return data


def _extract_main_output(model_output):
    if isinstance(model_output, dict):
        return model_output.get("main")
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def predict_smiles_list(model, cfg, smiles_list: List[str], task_names: List[str], target_task: str, batch_size: int) -> Dict[str, float]:
    try:
        from torch_geometric.loader import DataLoader
    except Exception:
        from torch_geometric.data import DataLoader

    unique_smiles, seen = [], set()
    for smi in smiles_list:
        can = canonicalize_smiles(smi)
        if can is None or can in seen:
            continue
        seen.add(can)
        unique_smiles.append(can)

    data_list, ok_smiles = [], []
    for smi in unique_smiles:
        data = mol_to_graph_data(smi, cfg)
        if data is not None:
            data_list.append(data)
            ok_smiles.append(smi)
    if not data_list:
        return {}

    task_idx = task_names.index(target_task) if target_task in task_names else 0
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds: Dict[str, float] = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = _extract_main_output(model(batch))
            if out is None:
                raise RuntimeError("Model output does not contain main tensor.")
            if cfg.DATA.TASK_TYPE == "classification":
                logits = out[:, task_idx * 2:(task_idx + 1) * 2]
                values = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            else:
                values = out[:, task_idx].detach().cpu().numpy()
            for j, val in enumerate(values):
                preds[ok_smiles[offset + j]] = float(val)
            offset += len(values)
    return preds


                                                                               
                  
                                                                               

def load_input_table(cfg, args: argparse.Namespace, task_names: List[str]) -> Tuple[pd.DataFrame, str, Optional[str]]:
    csv_path = args.input_csv or os.path.join(cfg.DATA.DATA_PATH, "raw", f"{cfg.DATA.DATASET}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column {args.smiles_col!r} not found in {csv_path}")

    target_task = args.target_task or (task_names[0] if task_names else None)
    if target_task is None:
        target_task = next(c for c in df.columns if c != args.smiles_col)
    true_col = target_task if target_task in df.columns else None
    if true_col is None:
        warnings.warn(f"Target task {target_task!r} not found in input CSV. Experimental ΔpKa will not be available.")

    df = df.copy()
    df["canonical_smiles"] = df[args.smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
                                                                          
                                                                              
                                                                               
                
    df.attrs["input_csv_path"] = csv_path
    return df, target_task, true_col


def build_true_value_map(df: pd.DataFrame, true_col: Optional[str]) -> Dict[str, float]:
    if true_col is None or true_col not in df.columns:
        return {}
    tmp = df[["canonical_smiles", true_col]].copy()
    tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
    tmp = tmp.dropna(subset=[true_col])
    return tmp.groupby("canonical_smiles")[true_col].mean().to_dict()


def _first_non_null_value(values: pd.Series) -> Any:
    vals = values.dropna()
    return vals.iloc[0] if len(vals) else np.nan


def _join_unique_values(values: pd.Series, precision: int = 10, max_values: int = 12) -> str:

    vals: List[str] = []
    for v in values.dropna().tolist():
        if isinstance(v, (float, int, np.floating, np.integer)):
            token = fmt_float(v, precision=precision)
        else:
            token = str(v)
        if token not in vals:
            vals.append(token)
        if len(vals) >= max_values:
            break
    return "|".join(vals)


def build_original_input_lookup(
    input_df: pd.DataFrame,
    args: argparse.Namespace,
    true_col: Optional[str],
) -> pd.DataFrame:


    cols = ["canonical_smiles"]
    if args.smiles_col in input_df.columns:
        cols.append(args.smiles_col)
    if true_col and true_col in input_df.columns:
        cols.append(true_col)
    if args.pka_type_col and args.pka_type_col in input_df.columns:
        cols.append(args.pka_type_col)

    work = input_df[cols].copy()
    work["_input_row_index"] = work.index.astype(int)
    rows: List[Dict[str, Any]] = []
    for can, g in work.groupby("canonical_smiles", dropna=False):
        rec: Dict[str, Any] = {
            "canonical_smiles": can,
            "original_source_rows": "|".join(str(int(i)) for i in g["_input_row_index"].tolist()),
            "original_n_rows": int(len(g)),
        }
        if args.smiles_col in g.columns:
            rec["original_smiles_first"] = _first_non_null_value(g[args.smiles_col])
            rec["original_smiles_values"] = _join_unique_values(g[args.smiles_col], max_values=12)
        if true_col and true_col in g.columns:
            pka_vals = pd.to_numeric(g[true_col], errors="coerce")
            rec["original_pka_col"] = true_col
            rec["original_pka_first"] = _first_non_null_value(pka_vals)
            rec["original_pka_mean"] = float(pka_vals.mean()) if pka_vals.notna().any() else np.nan
            rec["original_pka_values"] = _join_unique_values(pka_vals, precision=10, max_values=12)
        if args.pka_type_col and args.pka_type_col in g.columns:
            rec["original_pka_type_col"] = args.pka_type_col
            rec["original_pka_type_first"] = _first_non_null_value(g[args.pka_type_col])
            rec["original_pka_type_values"] = _join_unique_values(g[args.pka_type_col], max_values=12)
        rows.append(rec)
    return pd.DataFrame(rows)


def annotate_site_rows_with_original_input_data(
    site_df: pd.DataFrame,
    input_df: pd.DataFrame,
    args: argparse.Namespace,
    true_col: Optional[str],
    input_csv_path: str,
) -> pd.DataFrame:


    if site_df is None or site_df.empty:
        return site_df

    out = site_df.copy()
    out["raw_input_csv_path"] = input_csv_path
    out["input_smiles_col"] = args.smiles_col
    out["input_pka_col"] = true_col if true_col is not None else ""

    if "source_row" in out.columns:
        source_rows = pd.to_numeric(out["source_row"], errors="coerce")
        if args.smiles_col in input_df.columns:
            smi_map = input_df[args.smiles_col].to_dict()
            out["halogenated_original_smiles"] = source_rows.map(lambda i: smi_map.get(int(i), np.nan) if pd.notna(i) else np.nan)
        if true_col and true_col in input_df.columns:
            pka_map = pd.to_numeric(input_df[true_col], errors="coerce").to_dict()
            out["halogenated_original_pka"] = source_rows.map(lambda i: pka_map.get(int(i), np.nan) if pd.notna(i) else np.nan)
        if args.pka_type_col and args.pka_type_col in input_df.columns:
            type_map = input_df[args.pka_type_col].to_dict()
            out["halogenated_original_pka_type"] = source_rows.map(lambda i: type_map.get(int(i), np.nan) if pd.notna(i) else np.nan)

    lookup = build_original_input_lookup(input_df, args, true_col)
    if not lookup.empty and "masked_parent_smiles" in out.columns:
        parent_lookup = lookup.add_prefix("dehalogenated_parent_")
        out = out.merge(
            parent_lookup,
            left_on="masked_parent_smiles",
            right_on="dehalogenated_parent_canonical_smiles",
            how="left",
        )
                                                                     
        if "dehalogenated_parent_original_smiles_first" in out.columns:
            out["dehalogenated_parent_original_smiles"] = out["dehalogenated_parent_original_smiles_first"]
            out["pre_halogenation_original_smiles"] = out["dehalogenated_parent_original_smiles_first"]
        if "dehalogenated_parent_original_pka_mean" in out.columns:
            out["dehalogenated_parent_original_pka"] = out["dehalogenated_parent_original_pka_mean"]
            out["pre_halogenation_original_pka"] = out["dehalogenated_parent_original_pka_mean"]

    return out


def enrich_fit_predictions_with_site_metadata(pred_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:


    if pred_df is None or pred_df.empty or site_df is None or site_df.empty or "site_id" not in pred_df.columns:
        return pred_df

    preferred_meta_cols = [
        "site_id", "source_row", "raw_input_csv_path", "input_smiles_col", "input_pka_col",
        "halogenated_original_smiles", "halogenated_original_pka", "halogenated_original_pka_type",
        "dehalogenated_parent_original_smiles", "dehalogenated_parent_original_pka",
        "dehalogenated_parent_original_pka_first", "dehalogenated_parent_original_pka_values",
        "dehalogenated_parent_original_smiles_values", "dehalogenated_parent_original_source_rows",
        "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
        "pka_true_halogenated", "pka_true_parent_matched",
        "pka_pred_halogenated", "pka_pred_parent_masked",
        "delta_pka_pred", "delta_pka_true", "delta_error",
        "halogen_idx", "attached_idx", "attached_symbol", "attached_is_aromatic",
        "acid_type", "base_type", "target_center_type",
        "pka_mode_inferred", "target_hii_mode",
        "target_hii_total", "acid_hii_total_mean", "base_hii_total_mean",
    ] + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS + TARGET_COMPONENT_COLS + ACID_COMPONENT_COLS + BASE_COMPONENT_COLS

    meta_cols = [c for c in preferred_meta_cols if c in site_df.columns]
    if "site_id" not in meta_cols:
        return pred_df
    meta = site_df[meta_cols].drop_duplicates(subset=["site_id"]).copy()

    rename_map: Dict[str, str] = {}
    for c in meta.columns:
        if c == "site_id":
            continue
        if c in pred_df.columns:
            new_c = f"site_{c}"
            while new_c in pred_df.columns or new_c in meta.columns:
                new_c = f"site_{new_c}"
            rename_map[c] = new_c
    if rename_map:
        meta = meta.rename(columns=rename_map)

    return pred_df.merge(meta, on="site_id", how="left")


def _safe_float(v) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def compute_scaffold_rdkit_descriptors(parent_smiles: str) -> Dict[str, float]:


    nan_result = {c: np.nan for c in SCAFFOLD_DESCRIPTOR_COLS}
    if not isinstance(parent_smiles, str) or not parent_smiles.strip():
        return nan_result
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return nan_result
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    try:
        return {
            "scaffold_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "scaffold_MolLogP": float(Descriptors.MolLogP(mol)),
            "scaffold_BertzCT": float(Descriptors.BertzCT(mol)),
            "scaffold_HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
            "scaffold_HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
            "scaffold_HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
            "scaffold_RotatableBonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "scaffold_RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
            "scaffold_AromaticRingCount": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "scaffold_FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }
    except Exception:
        return nan_result


def add_scaffold_background_scores(site_df: pd.DataFrame) -> pd.DataFrame:


    if site_df is None or site_df.empty:
        return site_df
    out = site_df.copy()
    if "masked_parent_smiles" not in out.columns:
        for c in SCAFFOLD_DESCRIPTOR_COLS + SCAFFOLD_COMPONENT_COLS:
            out[c] = np.nan
        return out

    unique_parents = pd.Series(out["masked_parent_smiles"].dropna().astype(str).unique())
    desc_map = {smi: compute_scaffold_rdkit_descriptors(smi) for smi in unique_parents.tolist()}
    desc_rows = [desc_map.get(str(smi), {c: np.nan for c in SCAFFOLD_DESCRIPTOR_COLS}) for smi in out["masked_parent_smiles"].astype(str)]
    desc_df = pd.DataFrame(desc_rows, index=out.index)
    for c in SCAFFOLD_DESCRIPTOR_COLS:
        out[c] = pd.to_numeric(desc_df.get(c, np.nan), errors="coerce")

    zcols: Dict[str, str] = {}
    for c in SCAFFOLD_DESCRIPTOR_COLS:
        values = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = values.median(skipna=True)
        if not np.isfinite(med):
            med = 0.0
        values = values.fillna(float(med))
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std(ddof=0)) if len(values) else 0.0
        zc = f"z_{c}"
        if std <= 0 or not np.isfinite(std):
            out[zc] = 0.0
        else:
            out[zc] = (values - mean) / std
        zcols[c] = zc

    out["scaffold_polarity"] = (
        out[zcols["scaffold_TPSA"]]
        + out[zcols["scaffold_HBA"]]
        + out[zcols["scaffold_HBD"]]
        - out[zcols["scaffold_MolLogP"]]
    ) / 4.0
    out["scaffold_complexity"] = (
        out[zcols["scaffold_BertzCT"]]
        + 0.5 * out[zcols["scaffold_HeavyAtomCount"]]
        + 0.5 * out[zcols["scaffold_RotatableBonds"]]
        + 0.5 * out[zcols["scaffold_FractionCSP3"]]
    ) / 2.5
    out["scaffold_ring"] = (
        out[zcols["scaffold_RingCount"]]
        + out[zcols["scaffold_AromaticRingCount"]]
    ) / 2.0

    for c in SCAFFOLD_COMPONENT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_halogen_mask_sites(df: pd.DataFrame, args: argparse.Namespace, true_col: Optional[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    site_count = 0
    for mol_idx, row in df.iterrows():
        smi = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        sites = get_halogen_sites(mol)
        for site in sites:
            if args.only_aromatic_halogen and not site["attached_is_aromatic"]:
                continue
            parent = mask_halogen_to_parent_smiles(smi, site["halogen_idx"])
            parent_can = canonicalize_smiles(parent) if parent else None
            if parent_can is None or parent_can == smi:
                continue

            pka_value = _safe_float(row[true_col]) if true_col and true_col in row.index else None
            pka_type_value = row[args.pka_type_col] if args.pka_type_col and args.pka_type_col in row.index else None
            hii = compute_site_acid_base_hii(
                smi,
                int(site["halogen_idx"]),
                pka_value=pka_value,
                pka_mode=None,
                pka_type_value=pka_type_value,
                aggregation=args.hii_aggregation,
                ambiguous_policy=args.ambiguous_policy,
                weight_tau=args.hii_weight_tau,
            )

            out = {
                "site_id": site_count,
                "source_row": int(mol_idx),
                "halogenated_smiles": smi,
                "masked_parent_smiles": parent_can,
                "halogen": site["halogen"],
                "halogen_idx": site["halogen_idx"],
                "attached_idx": site["attached_idx"],
                "attached_symbol": site["attached_symbol"],
                "attached_is_aromatic": site["attached_is_aromatic"],
            }
            if true_col and true_col in row.index:
                out["pka_true_halogenated"] = pd.to_numeric(row[true_col], errors="coerce")
            if args.pka_type_col and args.pka_type_col in row.index:
                out["pka_type_raw"] = row[args.pka_type_col]
            out.update(hii)
            rows.append(out)
            site_count += 1
            if args.max_sites is not None and site_count >= args.max_sites:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


                                                                               
                       
                                                                               

def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    for c in cols:
        work[c] = _as_float_series(work[c])
    return work.dropna(subset=cols)


def build_estimator(kind: str):
    if kind == "linear":
        return LinearRegression()
    if kind == "huber":
        return HuberRegressor(max_iter=1000)
    if kind == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 17))
    raise ValueError(kind)


def get_cv(work: pd.DataFrame, args: argparse.Namespace):
    n = len(work)
    if args.cv == "none" or n < max(10, args.n_splits):
        return None, None
    if args.cv == "group" and args.group_col in work.columns:
        groups = work[args.group_col].astype(str).fillna("NA").values
        n_groups = len(np.unique(groups))
        if n_groups >= 2:
            return GroupKFold(n_splits=min(args.n_splits, n_groups)), groups
    return KFold(n_splits=min(args.n_splits, n), shuffle=True, random_state=2026), None


def metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    yt, yp = y[mask], yhat[mask]
    return {
        "n": int(mask.sum()),
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
    }


def fit_numeric_model(df: pd.DataFrame, target: str, x_cols: List[str], args: argparse.Namespace, subset_label: str, model_label: str) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    work = _clean(df, [target] + x_cols)
    rec: Dict[str, Any] = {
        "target": target,
        "subset": subset_label,
        "model": model_label,
        "n": len(work),
        "n_features": len(x_cols),
        "full_data_r2": np.nan,
        "full_data_rmse": np.nan,
        "full_data_mae": np.nan,
                                                                                 
                                                                     
        "train_r2": np.nan,
        "cv_r2": np.nan,
        "train_rmse": np.nan,
        "cv_rmse": np.nan,
        "train_mae": np.nan,
        "cv_mae": np.nan,
        "raw_intercept": np.nan,
        "formula": "",
        "x_cols": "|".join(x_cols),
        "final_formula_transform": FINAL_FORMULA_TRANSFORM,
        "final_formula_scale_divisor": float(FINAL_FORMULA_SCALE),
        "final_formula_offset": float(FINAL_FORMULA_OFFSET),
    }
    if len(work) < max(5, len(x_cols) + 2):
        return rec, None

    X = work[x_cols].astype(float).values
    y = work[target].astype(float).values
    pipe = Pipeline([("scale", StandardScaler()), ("reg", build_estimator(args.fit_model))])
    pipe.fit(X, y)

    yhat_before_final_transform = pipe.predict(X)
    yhat = apply_final_formula_transform(yhat_before_final_transform)
                                                                          
                                                                          
                                                                           
                                                 
    m = metrics(y, yhat_before_final_transform)
    rec.update({
        "full_data_r2": m["r2"],
        "full_data_rmse": m["rmse"],
        "full_data_mae": m["mae"],
        "train_r2": m["r2"],
        "train_rmse": m["rmse"],
        "train_mae": m["mae"],
    })

    pred_df = work[[c for c in ["site_id", "halogen", "halogenated_smiles", "masked_parent_smiles", "pka_mode_inferred", "target_hii_mode", target] if c in work.columns]].copy()
    pred_df["observed"] = y
    pred_df["fitted_full_data_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_full_data"] = yhat
    pred_df["fitted_train_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_train"] = yhat
    pred_df["fitted_cv_before_final_transform"] = np.nan
    pred_df["fitted_cv"] = np.nan
    pred_df["final_formula_transform"] = FINAL_FORMULA_TRANSFORM
    pred_df["final_formula_scale_divisor"] = float(FINAL_FORMULA_SCALE)
    pred_df["final_formula_offset"] = float(FINAL_FORMULA_OFFSET)
    pred_df["target"] = target
    pred_df["subset"] = subset_label
    pred_df["model"] = model_label

    cv, groups = get_cv(work, args)
    if cv is not None:
        try:
            if groups is not None:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
            else:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv)
            ycv = apply_final_formula_transform(ycv_before_final_transform)
                                                                            
                                                                           
            cm = metrics(y, ycv_before_final_transform)
            rec.update({"cv_r2": cm["r2"], "cv_rmse": cm["rmse"], "cv_mae": cm["mae"]})
            pred_df["fitted_cv_before_final_transform"] = ycv_before_final_transform
            pred_df["fitted_cv"] = ycv
        except Exception as exc:
            rec["cv_error"] = str(exc)

    reg = pipe.named_steps["reg"]
    scaler = pipe.named_steps["scale"]
    coef = getattr(reg, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef).ravel()
        raw_coef = coef / scaler.scale_
        raw_intercept = float(getattr(reg, "intercept_", 0.0) - np.sum(coef * scaler.mean_ / scaler.scale_))
        add_formula_to_record(rec, target, x_cols, raw_intercept, raw_coef, args)
    return rec, pred_df


def annotate_acid_partitions(df: pd.DataFrame) -> pd.DataFrame:


    out = df.copy()
    mode = out["pka_mode_inferred"].astype(str).str.lower() if "pka_mode_inferred" in out.columns else pd.Series(["unknown"] * len(out), index=out.index)
    acidic_mask = mode.eq("acidic") | mode.str.contains("fallback_acid", case=False, na=False)

    out["acid_unified"] = acidic_mask.astype(bool)
    out["acid_partition"] = np.where(acidic_mask.values, "acidic_unified", "non_acidic")
    out["acid_subpartition"] = out["acid_partition"]

                                                                             
                                                                              
    out["acid_high_conf_aromatic"] = False
    out["acid_non_aromatic"] = False
    out["acid_other_aromatic"] = False
    out["acid_remaining"] = acidic_mask.astype(bool)
    return out

def _safe_feature_token(value: Any) -> str:

    s = str(value) if value is not None else "NA"
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "NA"


_ACID_COMPONENT_SHORT = {
    "acid_hii_inductive_selected": "I",
    "acid_hii_resonance_selected": "R",
    "acid_hii_aromatic_selected": "A",
    "acid_hii_steric_selected": "S",
}


def scaffold_triple_model_label(scaffold_cols: List[str]) -> str:

    tokens = [str(c).replace("scaffold_", "") for c in scaffold_cols]
    return f"{SCAFFOLD_TRIPLE_MODEL_PREFIX}__" + "__".join(tokens)


def scaffold_triple_display(scaffold_cols: List[str]) -> str:

    return "|".join(str(c) for c in scaffold_cols)


def is_scaffold_triple_model(model: Any) -> bool:
    return str(model).startswith(f"{SCAFFOLD_TRIPLE_MODEL_PREFIX}__")


def model_formula_explanation(model: Any) -> str:
    model = str(model)
    if is_scaffold_triple_model(model):
        return MODEL_FORMULA_EXPLANATIONS.get(SCAFFOLD_TRIPLE_MODEL_PREFIX, "")
    return MODEL_FORMULA_EXPLANATIONS.get(model, "")


def build_acid_fixed_component_frame(
    work: pd.DataFrame,
    component_cols: List[str] = ACID_COMPONENT_COLS,
    scaffold_cols: List[str] = SCAFFOLD_COMPONENT_COLS,
) -> Tuple[pd.DataFrame, List[str]]:


    x_cols = list(component_cols) + list(scaffold_cols)
    missing = [c for c in x_cols if c not in work.columns]
    if missing:
        raise ValueError(f"Missing fixed acid formula feature columns: {missing}")
    X = work[x_cols].astype(float).copy()
    return X, x_cols


def fit_acid_fixed_components_model(
    df: pd.DataFrame,
    target: str,
    args: argparse.Namespace,
    subset_label: str = "acidic_only",
    scaffold_cols: Optional[List[str]] = None,
    model_label: str = "acid_fixed_components",
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:


    scaffold_cols = list(scaffold_cols or SCAFFOLD_COMPONENT_COLS)
    required = [target] + ACID_COMPONENT_COLS + scaffold_cols
    work = df.copy()
    if "n_acid_pairs" in work.columns:
        work = work[pd.to_numeric(work["n_acid_pairs"], errors="coerce").fillna(0) > 0].copy()
    for c in required:
        work[c] = _as_float_series(work[c])
    work = work.dropna(subset=required).copy()

    rec: Dict[str, Any] = {
        "target": target,
        "subset": subset_label,
        "model": model_label,
        "n": len(work),
        "n_features": np.nan,
        "full_data_r2": np.nan,
        "full_data_rmse": np.nan,
        "full_data_mae": np.nan,
        "train_r2": np.nan,
        "cv_r2": np.nan,
        "train_rmse": np.nan,
        "cv_rmse": np.nan,
        "train_mae": np.nan,
        "cv_mae": np.nan,
        "raw_intercept": np.nan,
        "formula": "",
        "x_cols": "|".join(ACID_COMPONENT_COLS + scaffold_cols),
        "fit_kind": "ridge_fixed_acid_components",
        "coefficient_scope": "global_fixed_one_set_for_all_halogens_and_acid_center_types_with_descriptor_triple",
        "scaffold_combo_size": len(scaffold_cols),
        "scaffold_descriptor_combo": scaffold_triple_display(scaffold_cols),
        "final_formula_transform": FINAL_FORMULA_TRANSFORM,
        "final_formula_scale_divisor": float(FINAL_FORMULA_SCALE),
        "final_formula_offset": float(FINAL_FORMULA_OFFSET),
    }
    min_n = max(int(getattr(args, "acid_fixed_min_n", 5)), 20)
    if len(work) < min_n:
        rec["cv_error"] = f"Not enough rows for fixed acid model: n={len(work)} < {min_n}"
        return rec, None

    Xdf, x_cols = build_acid_fixed_component_frame(work, scaffold_cols=scaffold_cols)
    rec.update({
        "n_features": len(x_cols),
        "x_cols": "|".join(x_cols),
    })
    X = Xdf.astype(float).values
    y = work[target].astype(float).values
    acid_fixed_alpha = float(getattr(args, "acid_fixed_alpha", 30.0))
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=acid_fixed_alpha)),
    ])
    pipe.fit(X, y)

    yhat_before_final_transform = pipe.predict(X)
    yhat = apply_final_formula_transform(yhat_before_final_transform)
                                                                          
                                                                          
                                                                           
                                                 
    m = metrics(y, yhat_before_final_transform)
    rec.update({
        "full_data_r2": m["r2"],
        "full_data_rmse": m["rmse"],
        "full_data_mae": m["mae"],
        "train_r2": m["r2"],
        "train_rmse": m["rmse"],
        "train_mae": m["mae"],
    })

    pred_cols = []
    for c in [
        "site_id", "halogen", "halogenated_smiles", "masked_parent_smiles",
        "pka_mode_inferred", "target_hii_mode", "acid_type", "acid_center_type", target,
    ] + ACID_COMPONENT_COLS + scaffold_cols + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS:
        if c in work.columns and c not in pred_cols:
            pred_cols.append(c)
    pred_df = work[pred_cols].copy()
    pred_df["observed"] = y
    pred_df["fitted_full_data_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_full_data"] = yhat
    pred_df["fitted_train_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_train"] = yhat
    pred_df["fitted_cv_before_final_transform"] = np.nan
    pred_df["fitted_cv"] = np.nan
    pred_df["final_formula_transform"] = FINAL_FORMULA_TRANSFORM
    pred_df["final_formula_scale_divisor"] = float(FINAL_FORMULA_SCALE)
    pred_df["final_formula_offset"] = float(FINAL_FORMULA_OFFSET)
    pred_df["target"] = target
    pred_df["subset"] = subset_label
    pred_df["model"] = model_label

    cv, groups = get_cv(work, args)
    if cv is not None:
        try:
            if groups is not None:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
            else:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv)
            ycv = apply_final_formula_transform(ycv_before_final_transform)
                                                                            
                                                                           
            cm = metrics(y, ycv_before_final_transform)
            rec.update({"cv_r2": cm["r2"], "cv_rmse": cm["rmse"], "cv_mae": cm["mae"]})
            pred_df["fitted_cv_before_final_transform"] = ycv_before_final_transform
            pred_df["fitted_cv"] = ycv
        except Exception as exc:
            rec["cv_error"] = str(exc)

    reg = pipe.named_steps["reg"]
    scaler = pipe.named_steps["scale"]
    coef = getattr(reg, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef).ravel()
        raw_coef = coef / scaler.scale_
        raw_intercept = float(getattr(reg, "intercept_", 0.0) - np.sum(coef * scaler.mean_ / scaler.scale_))
        rec["ridge_alpha"] = float(getattr(reg, "alpha", np.nan)) if getattr(reg, "alpha", None) is not None else np.nan
        add_formula_to_record(rec, target, x_cols, raw_intercept, raw_coef, args)
    return rec, pred_df


def fit_mixed_context_model(df: pd.DataFrame, target: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:


    num_cols = TARGET_COMPONENT_COLS
    cat_cols = ["pka_mode_inferred"]
    work = df.copy()
    for c in [target] + num_cols:
        work[c] = _as_float_series(work[c])
    work = work.dropna(subset=[target] + num_cols)
    work["pka_mode_inferred"] = work["pka_mode_inferred"].astype(str).fillna("unknown")

    rec: Dict[str, Any] = {
        "target": target,
        "subset": "all",
        "model": "target_components_plus_pka_mode_interactions",
        "n": len(work),
        "n_features": np.nan,
        "full_data_r2": np.nan,
        "full_data_rmse": np.nan,
        "full_data_mae": np.nan,
        "train_r2": np.nan,
        "cv_r2": np.nan,
        "train_rmse": np.nan,
        "cv_rmse": np.nan,
        "train_mae": np.nan,
        "cv_mae": np.nan,
        "raw_intercept": np.nan,
        "formula": "",
        "x_cols": "target_hii_components + pka_mode + interactions",
        "final_formula_transform": FINAL_FORMULA_TRANSFORM,
        "final_formula_scale_divisor": float(FINAL_FORMULA_SCALE),
        "final_formula_offset": float(FINAL_FORMULA_OFFSET),
    }
    if len(work) < 20:
        return rec, None

                                                                         
    Xnum = work[num_cols].astype(float).copy()
    mode = work["pka_mode_inferred"].astype(str)
    for m in ["acidic", "basic", "ambiguous"]:
        dummy = (mode == m).astype(float)
        Xnum[f"mode_{m}"] = dummy
        for c in num_cols:
            Xnum[f"{c}_x_{m}"] = Xnum[c] * dummy
    x_cols = list(Xnum.columns)
    rec["x_cols"] = "|".join(x_cols)
    X = Xnum.values
    y = work[target].astype(float).values
    pipe = Pipeline([("scale", StandardScaler()), ("reg", build_estimator(args.fit_model))])
    pipe.fit(X, y)

    yhat_before_final_transform = pipe.predict(X)
    yhat = apply_final_formula_transform(yhat_before_final_transform)
                                                                          
                                                                          
                                                                           
                                                 
    m = metrics(y, yhat_before_final_transform)
    rec.update({
        "n_features": len(x_cols),
        "full_data_r2": m["r2"],
        "full_data_rmse": m["rmse"],
        "full_data_mae": m["mae"],
        "train_r2": m["r2"],
        "train_rmse": m["rmse"],
        "train_mae": m["mae"],
    })

    pred_df = work[[c for c in ["site_id", "halogen", "halogenated_smiles", "masked_parent_smiles", "pka_mode_inferred", "target_hii_mode", target] if c in work.columns]].copy()
    pred_df["observed"] = y
    pred_df["fitted_full_data_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_full_data"] = yhat
    pred_df["fitted_train_before_final_transform"] = yhat_before_final_transform
    pred_df["fitted_train"] = yhat
    pred_df["fitted_cv_before_final_transform"] = np.nan
    pred_df["fitted_cv"] = np.nan
    pred_df["final_formula_transform"] = FINAL_FORMULA_TRANSFORM
    pred_df["final_formula_scale_divisor"] = float(FINAL_FORMULA_SCALE)
    pred_df["final_formula_offset"] = float(FINAL_FORMULA_OFFSET)
    pred_df["target"] = target
    pred_df["subset"] = "all"
    pred_df["model"] = rec["model"]

    cv, groups = get_cv(work, args)
    if cv is not None:
        try:
            if groups is not None:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
            else:
                ycv_before_final_transform = cross_val_predict(pipe, X, y, cv=cv)
            ycv = apply_final_formula_transform(ycv_before_final_transform)
                                                                            
                                                                           
            cm = metrics(y, ycv_before_final_transform)
            rec.update({"cv_r2": cm["r2"], "cv_rmse": cm["rmse"], "cv_mae": cm["mae"]})
            pred_df["fitted_cv_before_final_transform"] = ycv_before_final_transform
            pred_df["fitted_cv"] = ycv
        except Exception as exc:
            rec["cv_error"] = str(exc)

    reg = pipe.named_steps["reg"]
    scaler = pipe.named_steps["scale"]
    coef = getattr(reg, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef).ravel()
        raw_coef = coef / scaler.scale_
        raw_intercept = float(getattr(reg, "intercept_", 0.0) - np.sum(coef * scaler.mean_ / scaler.scale_))
        add_formula_to_record(rec, target, x_cols, raw_intercept, raw_coef, args)
    return rec, pred_df


def fit_all_models(site_df: pd.DataFrame, targets: List[str], args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:


    records: List[Dict[str, Any]] = []
    preds: List[pd.DataFrame] = []

    if bool(getattr(args, "disable_acid_fixed", False)):
        return pd.DataFrame(records), pd.DataFrame()

    mode = site_df["pka_mode_inferred"].astype(str).str.lower() if "pka_mode_inferred" in site_df.columns else pd.Series(["unknown"] * len(site_df), index=site_df.index)
    acidic_mask = mode.eq("acidic") | mode.str.contains("fallback_acid", case=False, na=False)
    acid_df = site_df[acidic_mask].copy()

    scaffold_triples = list(combinations(SCAFFOLD_TRIPLE_DESCRIPTOR_COLS, SCAFFOLD_TRIPLE_SIZE))

    for target in targets:
        if target not in acid_df.columns:
            continue
        if len(acid_df) < args.min_group_size:
            continue
        if not all(c in acid_df.columns for c in ACID_COMPONENT_COLS + SCAFFOLD_TRIPLE_DESCRIPTOR_COLS):
            continue
        if "halogen" not in acid_df.columns:
            continue

        for scaffold_cols_tuple in scaffold_triples:
            scaffold_cols = list(scaffold_cols_tuple)
            model_label = scaffold_triple_model_label(scaffold_cols)
            rec, p = fit_acid_fixed_components_model(
                acid_df,
                target,
                args,
                subset_label="acidic_only",
                scaffold_cols=scaffold_cols,
                model_label=model_label,
            )
            rec["scaffold_combo_size"] = SCAFFOLD_TRIPLE_SIZE
            rec["scaffold_descriptor_combo"] = scaffold_triple_display(scaffold_cols)
            records.append(rec)
            if p is not None:
                p["scaffold_combo_size"] = SCAFFOLD_TRIPLE_SIZE
                p["scaffold_descriptor_combo"] = scaffold_triple_display(scaffold_cols)
                preds.append(p)

    fit_df = pd.DataFrame(records)
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return fit_df, pred_df

def build_all_formula_table(fit_df: pd.DataFrame) -> pd.DataFrame:

    preferred = [
        "target", "subset", "model", "scaffold_combo_size", "scaffold_descriptor_combo",
        "n", "n_features",
        "full_data_r2", "full_data_rmse", "full_data_mae",
        "formula", "x_cols", "raw_intercept",
                                                                                 
        "train_r2", "cv_r2", "train_rmse", "cv_rmse", "train_mae", "cv_mae",
    ]
    if fit_df.empty:
        return pd.DataFrame(columns=preferred)
    out = fit_df.copy()
    if "full_data_r2" not in out.columns and "train_r2" in out.columns:
        out["full_data_r2"] = out["train_r2"]
    if "full_data_rmse" not in out.columns and "train_rmse" in out.columns:
        out["full_data_rmse"] = out["train_rmse"]
    if "full_data_mae" not in out.columns and "train_mae" in out.columns:
        out["full_data_mae"] = out["train_mae"]
    if "formula" not in out.columns:
        out["formula"] = ""
    show = [c for c in preferred if c in out.columns]
    coef_cols = [c for c in out.columns if c.startswith("coef_raw__")]
    out = out[show + coef_cols]
    sort_cols = [c for c in ["target", "subset", "full_data_r2"] if c in out.columns]
    if sort_cols and "full_data_r2" in sort_cols:
        out = out.sort_values(["target", "subset", "full_data_r2"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def select_final_formula_rows(fit_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:


    preferred = [
        "target", "subset", "model", "scaffold_combo_size", "scaffold_descriptor_combo",
        "n", "n_features", "full_data_r2",
        "full_data_rmse", "full_data_mae", "selection_rule", "formula", "x_cols",
    ]
    if fit_df.empty:
        empty = pd.DataFrame(columns=preferred)
        return empty.copy(), empty.copy()

    work = fit_df.copy()
    if "formula" not in work.columns:
        work["formula"] = ""
    if "full_data_r2" not in work.columns and "train_r2" in work.columns:
        work["full_data_r2"] = work["train_r2"]
    if "full_data_rmse" not in work.columns and "train_rmse" in work.columns:
        work["full_data_rmse"] = work["train_rmse"]
    if "full_data_mae" not in work.columns and "train_mae" in work.columns:
        work["full_data_mae"] = work["train_mae"]

    work["formula"] = work["formula"].fillna("").astype(str)
    work["full_data_r2"] = _as_float_series(work["full_data_r2"]) if "full_data_r2" in work.columns else np.nan
    work = work[(work["formula"].str.len() > 0) & work["full_data_r2"].notna()].copy()
    if work.empty:
        empty = pd.DataFrame(columns=preferred)
        return empty.copy(), empty.copy()

    selected_model = getattr(args, "main_formula_model", "auto_best_full_data")
    if selected_model != "auto_best_full_data":
        work = work[work["model"].astype(str) == selected_model].copy()
        selection_rule = f"selected_model:{selected_model}"
    else:
        selection_rule = "max_full_data_r2"
    if work.empty:
        empty = pd.DataFrame(columns=preferred)
        return empty.copy(), empty.copy()

    work["selection_rule"] = selection_rule
    work["_model_priority"] = work["model"].map(FINAL_FORMULA_MODEL_PRIORITY).fillna(99).astype(float)
    sort_cols = ["target", "subset", "full_data_r2", "_model_priority", "n"]
    work = work.sort_values(sort_cols, ascending=[True, True, False, True, False])
    final_by_subset = work.groupby(["target", "subset"], dropna=False, as_index=False).head(1).copy()

    global_candidates = work[work["subset"].astype(str) == "all"].copy()
    if global_candidates.empty:
        global_candidates = work.copy()
    global_candidates = global_candidates.sort_values(
        ["target", "full_data_r2", "_model_priority", "n"],
        ascending=[True, False, True, False],
    )
    final_global = global_candidates.groupby(["target"], dropna=False, as_index=False).head(1).copy()

    show = [c for c in preferred if c in final_global.columns]
    final_global = final_global[show].reset_index(drop=True)
    show_subset = [c for c in preferred if c in final_by_subset.columns]
    final_by_subset = final_by_subset[show_subset].reset_index(drop=True)
    return final_global, final_by_subset


def export_scaffold_triple_r2_ranking(fit_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:

    cols = [
        "target", "subset", "model", "scaffold_combo_size", "scaffold_descriptor_combo",
        "n", "n_features", "full_data_r2", "full_data_rmse", "full_data_mae",
        "train_r2", "cv_r2", "train_rmse", "cv_rmse", "train_mae", "cv_mae",
        "ridge_alpha", "formula", "x_cols",
    ]
    if fit_df is None or fit_df.empty or "model" not in fit_df.columns:
        out = pd.DataFrame(columns=cols + ["rank_full_data_r2", "rank_cv_r2"])
        out.to_csv(os.path.join(output_dir, "scaffold_triple_r2_ranking.csv"), index=False)
        return out

    out = fit_df[fit_df["model"].astype(str).map(is_scaffold_triple_model)].copy()
    if out.empty:
        out = pd.DataFrame(columns=cols + ["rank_full_data_r2", "rank_cv_r2"])
        out.to_csv(os.path.join(output_dir, "scaffold_triple_r2_ranking.csv"), index=False)
        return out

    for metric in ["full_data_r2", "cv_r2"]:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")

    out["rank_full_data_r2"] = out.groupby("target")["full_data_r2"].rank(method="min", ascending=False) if "full_data_r2" in out.columns else np.nan
    out["rank_cv_r2"] = out.groupby("target")["cv_r2"].rank(method="min", ascending=False) if "cv_r2" in out.columns else np.nan

    show = [c for c in cols if c in out.columns] + ["rank_full_data_r2", "rank_cv_r2"]
    coef_cols = [c for c in out.columns if str(c).startswith("coef_raw__")]
    out = out[show + coef_cols]
    sort_cols = [c for c in ["target", "rank_full_data_r2", "rank_cv_r2"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
    out.to_csv(os.path.join(output_dir, "scaffold_triple_r2_ranking.csv"), index=False)
    return out

def export_final_formula_outputs(fit_df: pd.DataFrame, output_dir: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    all_formula_df = build_all_formula_table(fit_df)
    final_global_df, final_by_subset_df = select_final_formula_rows(fit_df, args)
    all_formula_df.to_csv(os.path.join(output_dir, "all_formulas_full_data_r2.csv"), index=False)
    final_global_df.to_csv(os.path.join(output_dir, "final_formula_full_data_r2.csv"), index=False)
    final_by_subset_df.to_csv(os.path.join(output_dir, "final_formula_by_subset_full_data_r2.csv"), index=False)
    return all_formula_df, final_global_df, final_by_subset_df


def export_acid_fixed_coefficients(fit_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:


    rows: List[Dict[str, Any]] = []
    component_name = {
        "acid_hii_inductive_selected": "inductive",
        "acid_hii_resonance_selected": "resonance",
        "acid_hii_aromatic_selected": "aromatic",
        "acid_hii_steric_selected": "steric",
        "scaffold_polarity": "scaffold_polarity",
        "scaffold_complexity": "scaffold_complexity",
        "scaffold_ring": "scaffold_ring",
        "scaffold_TPSA": "TPSA",
        "scaffold_MolLogP": "MolLogP",
        "scaffold_BertzCT": "BertzCT",
        "scaffold_HeavyAtomCount": "HeavyAtomCount",
        "scaffold_HBA": "HBA",
        "scaffold_HBD": "HBD",
        "scaffold_RotatableBonds": "RotatableBonds",
        "scaffold_RingCount": "RingCount",
        "scaffold_AromaticRingCount": "AromaticRingCount",
        "scaffold_FractionCSP3": "FractionCSP3",
    }
    component_short = {
        "acid_hii_inductive_selected": "I",
        "acid_hii_resonance_selected": "R",
        "acid_hii_aromatic_selected": "A",
        "acid_hii_steric_selected": "S",
        "scaffold_polarity": "S_polar",
        "scaffold_complexity": "S_complexity",
        "scaffold_ring": "S_ring",
        "scaffold_TPSA": "TPSA",
        "scaffold_MolLogP": "MolLogP",
        "scaffold_BertzCT": "BertzCT",
        "scaffold_HeavyAtomCount": "HeavyAtomCount",
        "scaffold_HBA": "HBA",
        "scaffold_HBD": "HBD",
        "scaffold_RotatableBonds": "RotBonds",
        "scaffold_RingCount": "RingCount",
        "scaffold_AromaticRingCount": "AromRingCount",
        "scaffold_FractionCSP3": "FractionCSP3",
    }
    empty_cols = [
        "target", "subset", "model", "scaffold_combo_size", "scaffold_descriptor_combo",
        "n", "n_features", "full_data_r2",
        "full_data_rmse", "full_data_mae", "train_r2", "cv_r2", "train_rmse",
        "cv_rmse", "train_mae", "cv_mae", "raw_intercept", "ridge_alpha",
        "coefficient_scope", "component", "component_short", "display_name",
        "coef_raw", "source_column", "explanation",
    ]
    if fit_df is None or fit_df.empty or "model" not in fit_df.columns:
        out = pd.DataFrame(columns=empty_cols)
        out.to_csv(os.path.join(output_dir, "acid_fixed_coefficients.csv"), index=False)
        return out

    fixed_rows = fit_df[fit_df["model"].astype(str).map(lambda m: m == "acid_fixed_components" or is_scaffold_triple_model(m))].copy()
    for _, row in fixed_rows.iterrows():
        meta = {
            "target": row.get("target"),
            "subset": row.get("subset"),
            "model": row.get("model"),
            "scaffold_combo_size": row.get("scaffold_combo_size", np.nan),
            "scaffold_descriptor_combo": row.get("scaffold_descriptor_combo", ""),
            "n": row.get("n"),
            "n_features": row.get("n_features"),
            "full_data_r2": row.get("full_data_r2"),
            "full_data_rmse": row.get("full_data_rmse"),
            "full_data_mae": row.get("full_data_mae"),
            "train_r2": row.get("train_r2"),
            "cv_r2": row.get("cv_r2"),
            "train_rmse": row.get("train_rmse"),
            "cv_rmse": row.get("cv_rmse"),
            "train_mae": row.get("train_mae"),
            "cv_mae": row.get("cv_mae"),
            "raw_intercept": row.get("raw_intercept", np.nan),
            "ridge_alpha": row.get("ridge_alpha", np.nan),
            "coefficient_scope": row.get("coefficient_scope", "global_fixed_one_set_for_all_halogens_and_acid_center_types"),
        }
        x_cols = [c for c in str(row.get("x_cols", "")).split("|") if c and c != "nan"]
        if not x_cols:
            x_cols = ACID_COMPONENT_COLS + SCAFFOLD_TRIPLE_DESCRIPTOR_COLS
        for feature in x_cols:
            col = f"coef_raw__{feature}"
            if col not in row.index or pd.isna(row.get(col, np.nan)):
                continue
            rec = dict(meta)
            rec.update({
                "component": component_name.get(feature, feature),
                "component_short": component_short.get(feature, feature_label(feature)),
                "display_name": feature_label(feature),
                "coef_raw": float(row[col]),
                "source_column": col,
                "explanation": explain_feature(feature),
            })
            rows.append(rec)

    out = pd.DataFrame(rows, columns=empty_cols)
    if not out.empty:
        out = out.sort_values(["target", "model", "source_column"]).reset_index(drop=True)
    out.to_csv(os.path.join(output_dir, "acid_fixed_coefficients.csv"), index=False)
    return out

def export_acid_single_scalar_diagnostics(fit_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:


    if fit_df.empty or "model" not in fit_df.columns:
        out = pd.DataFrame()
        out.to_csv(os.path.join(output_dir, "acid_single_scalar_diagnostics.csv"), index=False)
        return out

    records: List[Dict[str, Any]] = []
    key_cols = ["target", "subset"]
    wanted = ["target_hii_total", "acid_directed_components", "target_hii_components", "acid_fixed_components"]
    base = fit_df[fit_df["model"].isin(wanted)].copy()
    for (target, subset), g in base.groupby(key_cols, dropna=False):
        rec: Dict[str, Any] = {"target": target, "subset": subset}
        for model in wanted:
            rows = g[g["model"] == model]
            if len(rows):
                r = rows.iloc[0]
                rec[f"n__{model}"] = r.get("n", np.nan)
                rec[f"cv_r2__{model}"] = r.get("cv_r2", np.nan)
                rec[f"train_r2__{model}"] = r.get("train_r2", np.nan)
                rec[f"cv_rmse__{model}"] = r.get("cv_rmse", np.nan)
                rec[f"cv_mae__{model}"] = r.get("cv_mae", np.nan)
        total = rec.get("cv_r2__target_hii_total", np.nan)
        comp = rec.get("cv_r2__acid_directed_components", np.nan)
        fixed = rec.get("cv_r2__acid_fixed_components", np.nan)
        if pd.notna(total) and pd.notna(comp):
            rec["remaining_global_scalar_gap_vs_four_components"] = float(comp - total)
        if pd.notna(total) and pd.notna(fixed):
            rec["fixed_components_gain_vs_global_scalar"] = float(fixed - total)
                                       
        gap = rec.get("remaining_global_scalar_gap_vs_four_components", np.nan)
        if pd.notna(gap):
            if gap <= 0.01:
                rec["single_scalar_status"] = "near_four_component_ceiling"
            elif gap <= 0.05:
                rec["single_scalar_status"] = "small_remaining_gap"
            else:
                rec["single_scalar_status"] = "single_scalar_limited_by_context_or_nonlinearity"
        records.append(rec)

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["target", "subset"]).reset_index(drop=True)
    out.to_csv(os.path.join(output_dir, "acid_single_scalar_diagnostics.csv"), index=False)
    return out

def _prediction_metrics_from_table(df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    if df is None or df.empty or pred_col not in df.columns or "observed" not in df.columns:
        return {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    work = df.dropna(subset=["observed", pred_col]).copy()
    return metrics(work["observed"].values, work[pred_col].values)


def export_acid_unified_r2_summary(fit_df: pd.DataFrame, pred_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:


    rows: List[Dict[str, Any]] = []
    if fit_df.empty or "target" not in fit_df.columns:
        out = pd.DataFrame()
        out.to_csv(os.path.join(output_dir, "acid_unified_r2_summary.csv"), index=False)
        out.to_csv(os.path.join(output_dir, "acid_tiered_r2_summary.csv"), index=False)
        return out

    targets = sorted([str(t) for t in fit_df["target"].dropna().unique().tolist()])
    wanted_models = [
        ("target_hii_total", "unified_target_scalar"),
        ("target_hii_components", "unified_target_components"),
        ("acid_directed_components", "unified_acid_four_components"),
        ("acid_fixed_components", "unified_acid_fixed_components"),
    ]

    for target in targets:
        for model, recommended_use in wanted_models:
            r = fit_df[
                (fit_df["target"].astype(str) == target)
                & (fit_df["subset"].astype(str) == "acidic_only")
                & (fit_df["model"].astype(str) == model)
            ]
            if len(r):
                rr = r.iloc[0]
                rows.append({
                    "target": target,
                    "acid_report": "unified_all_acidic",
                    "subset": "acidic_only",
                    "model": model,
                    "recommended_use": recommended_use,
                    "n": rr.get("n", np.nan),
                    "n_features": rr.get("n_features", np.nan),
                    "full_data_r2": rr.get("full_data_r2", rr.get("train_r2", np.nan)),
                    "full_data_rmse": rr.get("full_data_rmse", rr.get("train_rmse", np.nan)),
                    "full_data_mae": rr.get("full_data_mae", rr.get("train_mae", np.nan)),
                    "train_r2": rr.get("train_r2", np.nan),
                    "cv_r2": rr.get("cv_r2", np.nan),
                    "train_rmse": rr.get("train_rmse", np.nan),
                    "cv_rmse": rr.get("cv_rmse", np.nan),
                    "train_mae": rr.get("train_mae", np.nan),
                    "cv_mae": rr.get("cv_mae", np.nan),
                    "note": "single unified acidic subset; no aromatic/non-aromatic split",
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_metric = "full_data_r2" if "full_data_r2" in out.columns else "cv_r2"
        out = out.sort_values(["target", sort_metric], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(os.path.join(output_dir, "acid_unified_r2_summary.csv"), index=False)
                                                                    
    out.to_csv(os.path.join(output_dir, "acid_tiered_r2_summary.csv"), index=False)
    return out


def export_acid_partition_summary(site_df: pd.DataFrame, output_dir: str, targets: List[str]) -> pd.DataFrame:

    if "acid_partition" not in site_df.columns:
        out = pd.DataFrame()
        out.to_csv(os.path.join(output_dir, "acid_partition_summary.csv"), index=False)
        return out
    rows: List[Dict[str, Any]] = []
    for partition, sub in site_df.groupby("acid_partition", dropna=False):
        rec: Dict[str, Any] = {"acid_partition": partition, "n_sites": len(sub)}
        for target in targets:
            if target in sub.columns:
                vals = _as_float_series(sub[target]).dropna()
                rec[f"{target}_n"] = int(len(vals))
                rec[f"{target}_mean"] = float(vals.mean()) if len(vals) else np.nan
                rec[f"{target}_median"] = float(vals.median()) if len(vals) else np.nan
                rec[f"{target}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("acid_partition").reset_index(drop=True)
    out.to_csv(os.path.join(output_dir, "acid_partition_summary.csv"), index=False)
    return out


def save_group_summaries(site_df: pd.DataFrame, output_dir: str, args: argparse.Namespace) -> None:
    group_cols = [
        "pka_mode_inferred", "target_hii_mode", "acid_partition", "acid_subpartition",
        "acid_unified", "halogen", "position", "target_position", "target_hii_position",
        "acid_position", "base_position", "acid_type", "base_type", "target_center_type",
        "attached_is_aromatic",
    ]
    target_cols = [c for c in [
        "delta_pka_pred", "delta_pka_true", "delta_error", "target_hii_total",
        "acid_hii_total_mean", "base_hii_total_mean"
    ] if c in site_df.columns]
    all_rows = []
    for group_col in group_cols:
        if group_col not in site_df.columns:
            continue
        rows = []
        for g, sub in site_df.groupby(group_col, dropna=False):
            if len(sub) < args.min_group_size:
                continue
            rec = {"group_col": group_col, "group": g, "n": len(sub)}
            for t in target_cols:
                vals = _as_float_series(sub[t]).dropna()
                if len(vals):
                    rec[f"{t}_mean"] = float(vals.mean())
                    rec[f"{t}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
                    rec[f"{t}_median"] = float(vals.median())
            rows.append(rec)
            all_rows.append(rec)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"group_summary_by_{group_col}.csv"), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(output_dir, "group_summary_all.csv"), index=False)


                                                                               
       
                                                                               

def fig_dir(output: str) -> str:
    p = os.path.join(output, "figures")
    os.makedirs(p, exist_ok=True)
    return p


def _svg_path_for_png(png_path: str) -> str:

    root, _ = os.path.splitext(str(png_path))
    return f"{root}.svg"


def save_figure_with_svg(fig, png_path: str, bbox_inches: str = "tight") -> str:


    png_path = str(png_path)
    svg_path = _svg_path_for_png(png_path)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=FIGURE_EXPORT_DPI, bbox_inches=bbox_inches)
    fig.savefig(svg_path, format="svg", bbox_inches=bbox_inches)
    return svg_path


                                                                             
                                                                            
                                                                           
FIT_PREDICTION_PLOT_SERIES = [
    ("fitted_full_data", "full_data"),
    ("fitted_train", "train"),
    ("fitted_cv", "cv"),
]

                                                                               
                                                                                 
                                                                              
               
FIGURE_SOURCE_ROWS: List[Dict[str, Any]] = []
FIGURE_SOURCE_MANIFEST = "figure_source_data_map.csv"


def reset_figure_source_manifest() -> None:
    FIGURE_SOURCE_ROWS.clear()


def _figure_relpath(output: str, figure_path: str) -> str:
    try:
        return os.path.relpath(figure_path, output)
    except Exception:
        return str(figure_path)


def _infer_output_dir_from_figure_path(figure_path: str) -> str:

    fig_parent = os.path.dirname(os.path.abspath(figure_path))
    if os.path.basename(fig_parent) == "figures":
        return os.path.dirname(fig_parent)
    return fig_parent


def _pipe_join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple, set)):
        return "|".join(str(v) for v in values)
    return str(values)


def register_figure_source(
    output: str,
    figure_path: str,
    source_csv: Any,
    source_columns: Any = None,
    plot_type: str = "",
    target: str = "",
    subset: str = "",
    model: str = "",
    fit_kind: str = "",
    exact_source_rows_csv: str = "",
    note: str = "",
) -> None:

    FIGURE_SOURCE_ROWS.append({
        "figure_file": _figure_relpath(output, figure_path),
        "svg_figure_file": _figure_relpath(output, _svg_path_for_png(figure_path)),
        "source_csv": _pipe_join(source_csv),
        "exact_source_rows_csv": str(exact_source_rows_csv) if exact_source_rows_csv is not None else "",
        "source_columns": _pipe_join(source_columns),
        "plot_type": plot_type,
        "target": str(target) if target is not None else "",
        "subset": str(subset) if subset is not None else "",
        "model": str(model) if model is not None else "",
        "fit_kind": str(fit_kind) if fit_kind is not None else "",
        "note": str(note) if note is not None else "",
    })


def write_figure_source_manifest(output: str) -> pd.DataFrame:

    cols = [
        "figure_file", "svg_figure_file", "source_csv", "exact_source_rows_csv",
        "source_columns", "plot_type", "target", "subset", "model",
        "fit_kind", "note",
    ]
    out_path = os.path.join(output, FIGURE_SOURCE_MANIFEST)
    if not FIGURE_SOURCE_ROWS:
        out = pd.DataFrame(columns=cols)
        out.to_csv(out_path, index=False)
        return out

    out = pd.DataFrame(FIGURE_SOURCE_ROWS)
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].drop_duplicates().sort_values(["figure_file", "source_csv"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def export_best_parity_source_rows(
    plot_df: pd.DataFrame,
    output: str,
    target: str,
    fit_kind: str,
    subset: str,
    model: str,
    pred_col: str,
    figure_path: str,
) -> str:


    if plot_df is None or plot_df.empty:
        return ""

                                                                           
                                                               
    out = prepare_unique_full_data_plot_rows(
        plot_df=plot_df,
        target=target,
        pred_col=pred_col,
        keep_distinct_observed_values=True,
    ).reset_index(drop=True)
    out.insert(0, "source_row_index_in_plot", np.arange(len(out), dtype=int))
    out.insert(1, "source_figure_file", _figure_relpath(output, figure_path))
    out.insert(2, "source_filter_target", str(target))
    out.insert(3, "source_filter_subset", str(subset))
    out.insert(4, "source_filter_model", str(model))
    out.insert(5, "source_fit_kind", str(fit_kind))
    out.insert(6, "plot_x_column", "observed")
    out.insert(7, "plot_y_column", str(pred_col))
    if pred_col in out.columns and "observed" in out.columns:
        out[f"residual_{fit_kind}"] = pd.to_numeric(out[pred_col], errors="coerce") - pd.to_numeric(out["observed"], errors="coerce")

    preferred_cols = [
        "source_row_index_in_plot", "source_figure_file", "source_filter_target",
        "source_filter_subset", "source_filter_model", "source_fit_kind",
        "plot_x_column", "plot_y_column", "observed", pred_col, f"residual_{fit_kind}",
        "target", "subset", "model",
        "formula_source_target", "formula_source_subset", "formula_source_model",
        "formula_source_fit_kind", "formula_source_selection_metric",
        "formula_training_observed_delta_pka_pred", "experimental_observed_delta_pka_true",
        "applied_formula_note",
        "halogen", "site_id", "source_row",
        "raw_input_csv_path", "input_smiles_col", "input_pka_col",
        "halogenated_original_smiles", "halogenated_original_pka", "halogenated_original_pka_type",
        "halogenated_smiles", "masked_parent_smiles",
        "dehalogenated_parent_original_smiles", "dehalogenated_parent_original_pka",
        "dehalogenated_parent_original_pka_first", "dehalogenated_parent_original_pka_values",
        "dehalogenated_parent_original_smiles_values", "dehalogenated_parent_original_source_rows",
        "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
        "pka_true_halogenated", "pka_true_parent_matched",
        "pka_pred_halogenated", "pka_pred_parent_masked",
        "delta_pka_pred", "delta_pka_true", "delta_error",
        "site_delta_pka_pred", "site_delta_pka_true", "site_delta_error",
        "pka_mode_inferred", "target_hii_mode", "acid_type", "acid_fixed",
        "target_hii_total", "acid_hii_total_mean", "base_hii_total_mean",
    ] + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS + TARGET_COMPONENT_COLS + ACID_COMPONENT_COLS + BASE_COMPONENT_COLS

    ordered_cols = [c for c in preferred_cols if c in out.columns]
    ordered_cols += [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols]

    target_token = _safe_feature_token(str(target)).lower()
    fit_token = _safe_feature_token(str(fit_kind)).lower()
    file_name = f"best_parity_{target_token}_{fit_token}_source_rows.csv"
    out.to_csv(os.path.join(output, file_name), index=False)
    return file_name


def plot_delta_vs_hii(site_df: pd.DataFrame, target: str, hii_col: str, out_path: str, title: str) -> None:
    if target not in site_df.columns or hii_col not in site_df.columns:
        return
    work = _clean(site_df, [target, hii_col])
    if len(work) < 3:
        return
    x = work[hii_col].astype(float).values.reshape(-1, 1)
    y = work[target].astype(float).values
    lr = LinearRegression().fit(x, y)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if "halogen" in work.columns:
        for h in HALOGEN_ORDER:
            sub = work[work["halogen"] == h]
            if len(sub):
                ax.scatter(sub[hii_col], sub[target], s=20, alpha=0.5, label=h)
    else:
        ax.scatter(work[hii_col], work[target], s=20, alpha=0.5)
    ax.plot(xs.ravel(), lr.predict(xs), linestyle="--", linewidth=2)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xlabel(hii_col)
    ax.set_ylabel(target)
    ax.set_title(f"{title}\nR²={r2_score(y, lr.predict(x)):.3f}, slope={float(lr.coef_[0]):.3f}, n={len(work)}")
    if "halogen" in work.columns:
        ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure_with_svg(fig, out_path)
    register_figure_source(
        output=_infer_output_dir_from_figure_path(out_path),
        figure_path=out_path,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=[target, hii_col, "halogen"],
        plot_type="delta_vs_hii",
        target=target,
        note="Rows are filtered to finite target and HII values before fitting the plotted line.",
    )
    plt.close(fig)


def parity_limits(a, b):
    vals = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -1, 1
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo) if not math.isclose(lo, hi) else 0.5
    return lo - pad, hi + pad


def _round_key_series(series: pd.Series, ndigits: int = 12) -> pd.Series:

    values = pd.to_numeric(series, errors="coerce")
    return values.round(ndigits).astype(str).fillna("NA")


def _join_unique_tokens(series: pd.Series, max_items: int = 32) -> str:

    seen: List[str] = []
    for value in series.dropna().tolist():
        token = str(value)
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_items:
            break
    return "|".join(seen)


def _counterfactual_dedup_key_columns(df: pd.DataFrame) -> List[str]:


    candidates = [
        "halogenated_smiles",
        "masked_parent_smiles",
        "halogen",
        "halogen_idx",
        "attached_idx",
        "target_center_type",
        "acid_type",
        "pka_mode_inferred",
    ]
    keys = [c for c in candidates if c in df.columns]
    if not keys:
        fallback = [c for c in ["halogenated_smiles", "masked_parent_smiles", "halogen"] if c in df.columns]
        keys = fallback
    return keys


def prepare_unique_full_data_plot_rows(
    plot_df: pd.DataFrame,
    target: str,
    pred_col: str,
    keep_distinct_observed_values: bool = True,
) -> pd.DataFrame:


    if plot_df is None or plot_df.empty:
        return plot_df

    out = plot_df.copy()
                                                                           
                                                           
    for existing in ["unique_full_data_row", "source_row_index_in_plot"]:
        if existing in out.columns:
            out = out.drop(columns=[existing])
    target = str(target)
    pred_col = str(pred_col)

    if "observed" in out.columns:
        out["true_value"] = pd.to_numeric(out["observed"], errors="coerce")
    if pred_col in out.columns:
        out["predicted_value"] = pd.to_numeric(out[pred_col], errors="coerce")

    if target == "delta_pka_pred":
        if "observed" in out.columns:
            out["observed_delta_pka_pred"] = pd.to_numeric(out["observed"], errors="coerce")
        if pred_col in out.columns:
            out["final_formula_pred_delta_pka_pred"] = pd.to_numeric(out[pred_col], errors="coerce")
    elif target == "delta_pka_true":
        if "observed" in out.columns:
            out["observed_delta_pka_true"] = pd.to_numeric(out["observed"], errors="coerce")
        if pred_col in out.columns:
            out["final_formula_pred_delta_pka_true"] = pd.to_numeric(out[pred_col], errors="coerce")

                                                                                
                                                                          
    out["full_data_definition"] = (
        "final fitted HII formula after /0.64109 + 0.210521 applied to unique "
        "acidic counterfactual records; not a train/validation/test/CV split"
    )
    out["full_data_formula_transform"] = FINAL_FORMULA_TRANSFORM
    out["full_data_dedup_requested_rule"] = "drop duplicates with same SMILES and same delta_pka"

    smiles_col = None
    for c in ["halogenated_smiles", "smiles", "canonical_smiles"]:
        if c in out.columns:
            smiles_col = c
            break

    key_cols: List[str] = []
    if smiles_col is not None and keep_distinct_observed_values and "observed" in out.columns:
        out["_delta_pka_dedup_key"] = _round_key_series(out["observed"])
        key_cols = [smiles_col, "_delta_pka_dedup_key"]
        out["full_data_dedup_key_columns"] = f"{smiles_col}|observed_delta_pka_rounded"
    else:
        key_cols = _counterfactual_dedup_key_columns(out)
        if keep_distinct_observed_values and "observed" in out.columns:
            out["_observed_dedup_key"] = _round_key_series(out["observed"])
            key_cols = key_cols + ["_observed_dedup_key"]
        out["full_data_dedup_key_columns"] = "|".join(key_cols)

    if key_cols:
        sort_cols = [c for c in ["source_row", "site_id"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, kind="mergesort")

        grouped = out.groupby(key_cols, dropna=False, sort=False)
        out["duplicate_count_before_dedup"] = grouped[key_cols[0]].transform("size")
        if "site_id" in out.columns:
            out["duplicate_site_ids"] = grouped["site_id"].transform(_join_unique_tokens)
        if "source_row" in out.columns:
            out["duplicate_source_rows"] = grouped["source_row"].transform(_join_unique_tokens)
        if "halogenated_original_pka" in out.columns:
            out["duplicate_halogenated_original_pka_values"] = grouped["halogenated_original_pka"].transform(_join_unique_tokens)
        if "dehalogenated_parent_original_pka" in out.columns:
            out["duplicate_parent_original_pka_values"] = grouped["dehalogenated_parent_original_pka"].transform(_join_unique_tokens)

        out = out.drop_duplicates(subset=key_cols, keep="first").copy()

    for tmp_col in ["_observed_dedup_key", "_delta_pka_dedup_key"]:
        if tmp_col in out.columns:
            out = out.drop(columns=[tmp_col])
    out = out.reset_index(drop=True)
    out.insert(0, "unique_full_data_row", np.arange(len(out), dtype=int))
    return out


def _best_parity_selection_metric(fit_df: pd.DataFrame) -> Optional[str]:

    selection_metric = "full_data_r2"
    if selection_metric not in fit_df.columns or fit_df[selection_metric].notna().sum() == 0:
        selection_metric = "cv_r2"
    if selection_metric not in fit_df.columns:
        return None
    return selection_metric


def _select_best_parity_fit_row(
    fit_df: pd.DataFrame,
    target: str,
    selection_metric: Optional[str] = None,
) -> Tuple[Optional[pd.Series], Optional[str]]:


    if fit_df is None or fit_df.empty or "target" not in fit_df.columns:
        return None, selection_metric

    selection_metric = selection_metric or _best_parity_selection_metric(fit_df)
    if selection_metric is None or selection_metric not in fit_df.columns:
        return None, selection_metric

    subfit = fit_df[(fit_df["target"].astype(str) == str(target)) & fit_df[selection_metric].notna()].copy()
    if subfit.empty:
        return None, selection_metric

    best_metric = subfit[selection_metric].max()
    candidates = subfit[subfit[selection_metric] >= best_metric - 0.01].copy()
    priority = {
        "acid_fixed_components": 0,
        "target_hii_components": 1,
        "acid_directed_components": 2,
        "target_components_plus_pka_mode_interactions": 3,
        "target_hii_total": 4,
    }
    candidates["prio"] = candidates["model"].map(priority).fillna(9)
    best = candidates.sort_values(["prio", selection_metric], ascending=[True, False]).iloc[0]
    return best, selection_metric


def _find_experimental_delta_column(work: pd.DataFrame) -> Optional[str]:

    for c in ["site_delta_pka_true", "delta_pka_true"]:
        if c in work.columns and pd.to_numeric(work[c], errors="coerce").notna().any():
            return c
    return None


def build_pred_formula_applied_to_true_table(
    pred_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    selection_metric: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[str], Dict[str, Any]]:


    empty_summary = {
        "formula_source": "delta_pka_pred",
        "evaluation_target": "delta_pka_true",
        "source_model": "",
        "source_subset": "",
        "source_fit_kind": "full_data",
        "source_selection_metric": selection_metric or "",
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "n": 0,
        "note": "delta_pka_pred formula applied to experimental delta_pka_true without refit",
    }
    if pred_df is None or pred_df.empty or "target" not in pred_df.columns:
        return pd.DataFrame(), None, selection_metric, empty_summary

    source_best, used_metric = _select_best_parity_fit_row(
        fit_df=fit_df,
        target="delta_pka_pred",
        selection_metric=selection_metric,
    )
    empty_summary["source_selection_metric"] = str(used_metric or "")
    if source_best is None:
        return pd.DataFrame(), None, used_metric, empty_summary

    if "fitted_full_data" not in pred_df.columns:
        return pd.DataFrame(), source_best, used_metric, empty_summary

    formula_work = pred_df[
        (pred_df["target"].astype(str) == "delta_pka_pred")
        & (pred_df["subset"].astype(str) == str(source_best.get("subset", "")))
        & (pred_df["model"].astype(str) == str(source_best.get("model", "")))
    ].copy()
    if formula_work.empty:
        return pd.DataFrame(), source_best, used_metric, empty_summary

    exp_col = _find_experimental_delta_column(formula_work)
    if exp_col is None:
        return pd.DataFrame(), source_best, used_metric, empty_summary

    w = formula_work.copy()
    w["formula_source_target"] = "delta_pka_pred"
    w["formula_source_subset"] = str(source_best.get("subset", ""))
    w["formula_source_model"] = str(source_best.get("model", ""))
    w["formula_source_fit_kind"] = "full_data"
    w["formula_source_selection_metric"] = str(used_metric or "")
    w["formula_training_observed_delta_pka_pred"] = pd.to_numeric(w["observed"], errors="coerce")
    w["experimental_observed_delta_pka_true"] = pd.to_numeric(w[exp_col], errors="coerce")
    w["observed"] = w["experimental_observed_delta_pka_true"]
    w["target"] = "delta_pka_true"
    w["subset"] = str(source_best.get("subset", ""))
    w["model"] = str(source_best.get("model", ""))
    w["applied_formula_note"] = (
        "fitted_full_data comes from the best delta_pka_pred HII formula; "
        "observed is experimental delta_pka_true; no refit on delta_pka_true."
    )
    w = w.dropna(subset=["observed", "fitted_full_data"]).copy()
    w = prepare_unique_full_data_plot_rows(
        plot_df=w,
        target="delta_pka_true",
        pred_col="fitted_full_data",
        keep_distinct_observed_values=True,
    )
    if w.empty:
        return w, source_best, used_metric, empty_summary

    m = metrics(w["observed"].values, w["fitted_full_data"].values)
    summary = {
        "formula_source": "delta_pka_pred",
        "evaluation_target": "delta_pka_true",
        "source_model": str(source_best.get("model", "")),
        "source_subset": str(source_best.get("subset", "")),
        "source_fit_kind": "full_data",
        "source_selection_metric": str(used_metric or ""),
        "r2": m["r2"],
        "rmse": m["rmse"],
        "mae": m["mae"],
        "n": int(m["n"]),
        "note": "delta_pka_pred formula applied to experimental delta_pka_true without refit",
    }
    return w, source_best, used_metric, summary


def _filter_acidic_full_data_rows(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return df
    mode_col = None
    for c in ["pka_mode_inferred", "site_pka_mode_inferred"]:
        if c in df.columns:
            mode_col = c
            break
    if mode_col is None:
        return df
    mode = df[mode_col].astype(str).str.lower()
    mask = mode.eq("acidic") | mode.str.contains("fallback_acid", case=False, na=False)
    return df[mask].copy()


def _order_full_data_columns(df: pd.DataFrame, pred_col: str = "fitted_full_data") -> pd.DataFrame:

    if df is None or df.empty:
        return df
    out = df.copy()
    if "halogenated_smiles" in out.columns and "smiles" not in out.columns:
        out["smiles"] = out["halogenated_smiles"]
    if "observed" in out.columns and "delta_pka" not in out.columns:
        out["delta_pka"] = pd.to_numeric(out["observed"], errors="coerce")
    if pred_col in out.columns and "final_formula_delta_pka" not in out.columns:
        out["final_formula_delta_pka"] = pd.to_numeric(out[pred_col], errors="coerce")
    if pred_col in out.columns and "observed" in out.columns:
        out["final_formula_residual"] = (
            pd.to_numeric(out[pred_col], errors="coerce")
            - pd.to_numeric(out["observed"], errors="coerce")
        )

    preferred = [
        "unique_full_data_row", "smiles", "delta_pka", "final_formula_delta_pka",
        "final_formula_residual", "observed", pred_col,
        "fitted_full_data_before_final_transform", "final_formula_transform",
        "full_data_dedup_requested_rule", "full_data_dedup_key_columns",
        "duplicate_count_before_dedup", "duplicate_site_ids", "duplicate_source_rows",
        "target", "subset", "model",
        "formula_source_target", "formula_source_subset", "formula_source_model",
        "formula_source_fit_kind", "formula_source_selection_metric",
        "formula_training_observed_delta_pka_pred", "experimental_observed_delta_pka_true",
        "halogen", "site_id", "source_row",
        "raw_input_csv_path", "input_smiles_col", "input_pka_col",
        "halogenated_smiles", "masked_parent_smiles",
        "halogenated_original_smiles", "halogenated_original_pka", "halogenated_original_pka_type",
        "dehalogenated_parent_original_smiles", "dehalogenated_parent_original_pka",
        "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
        "pka_true_halogenated", "pka_true_parent_matched",
        "pka_pred_halogenated", "pka_pred_parent_masked",
        "delta_pka_pred", "delta_pka_true", "delta_error",
        "site_delta_pka_pred", "site_delta_pka_true", "site_delta_error",
        "pka_mode_inferred", "target_hii_mode", "acid_type", "target_center_type",
        "target_hii_total", "acid_hii_total_mean", "base_hii_total_mean",
    ] + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS + TARGET_COMPONENT_COLS + ACID_COMPONENT_COLS + BASE_COMPONENT_COLS
    ordered = [c for c in preferred if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered]


def export_acidic_final_formula_full_data_tables(
    pred_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output: str,
) -> pd.DataFrame:


    summary_rows: List[Dict[str, Any]] = []
    if pred_df is None or pred_df.empty or fit_df is None or fit_df.empty:
        summary = pd.DataFrame([{
            "table": "pred_full_data",
            "n": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "note": "missing prediction or fit table",
        }])
        summary.to_csv(os.path.join(output, FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV), index=False)
        return summary

    source_best, used_metric = _select_best_parity_fit_row(
        fit_df=fit_df,
        target="delta_pka_pred",
        selection_metric=_best_parity_selection_metric(fit_df),
    )
    if source_best is None:
        summary = pd.DataFrame([{
            "table": "pred_full_data",
            "n": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "note": "no fitted delta_pka_pred final formula found",
        }])
        summary.to_csv(os.path.join(output, FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV), index=False)
        return summary

    source_subset = str(source_best.get("subset", ""))
    source_model = str(source_best.get("model", ""))

    pred_work = pred_df[
        (pred_df["target"].astype(str) == "delta_pka_pred")
        & (pred_df["subset"].astype(str) == source_subset)
        & (pred_df["model"].astype(str) == source_model)
    ].copy()
    pred_work = _filter_acidic_full_data_rows(pred_work)
    pred_work = pred_work.dropna(subset=["observed", "fitted_full_data"]).copy()
    pred_full_data = prepare_unique_full_data_plot_rows(
        plot_df=pred_work,
        target="delta_pka_pred",
        pred_col="fitted_full_data",
        keep_distinct_observed_values=True,
    )
    pred_full_data["formula_source_target"] = "delta_pka_pred"
    pred_full_data["formula_source_subset"] = source_subset
    pred_full_data["formula_source_model"] = source_model
    pred_full_data["formula_source_fit_kind"] = "full_data"
    pred_full_data["formula_source_selection_metric"] = str(used_metric or "")
    pred_full_data = _order_full_data_columns(pred_full_data, pred_col="fitted_full_data")

    pred_full_data.to_csv(os.path.join(output, ACIDIC_PRED_FULL_DATA_CSV), index=False)
    pred_full_data.to_csv(os.path.join(output, PRED_FULL_DATA_CSV), index=False)
    pm_final = metrics(pred_full_data["observed"].values, pred_full_data["fitted_full_data"].values) if len(pred_full_data) else {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
    before_col = "fitted_full_data_before_final_transform"
    pm_before = metrics(pred_full_data["observed"].values, pred_full_data[before_col].values) if len(pred_full_data) and before_col in pred_full_data.columns else pm_final
    unchanged_r2 = source_best.get("full_data_r2", pm_before["r2"])
    summary_rows.append({
        "table": "pred_full_data",
        "csv": PRED_FULL_DATA_CSV,
        "alias_csv": ACIDIC_PRED_FULL_DATA_CSV,
        "evaluation_target": "delta_pka_pred",
        "formula_source": "delta_pka_pred",
        "source_subset": source_subset,
        "source_model": source_model,
        "source_selection_metric": str(used_metric or ""),
        "final_formula_transform": FINAL_FORMULA_TRANSFORM,
        "dedup_rule": "same SMILES + same delta_pka",
        "n": int(pm_final["n"]),
                                                                                 
                                                                             
        "r2": unchanged_r2,
        "r2_source": "original_fitted_formula_before_postfit_transform",
        "final_formula_r2_recomputed_against_observed": pm_final["r2"],
        "rmse": pm_final["rmse"],
        "mae": pm_final["mae"],
        "rmse_before_final_transform": pm_before["rmse"],
        "mae_before_final_transform": pm_before["mae"],
        "note": "acidic model-mask full_data after final formula transform; R2 kept from original fitted formula",
    })

    exp_col = _find_experimental_delta_column(pred_work)
    if exp_col is not None:
        true_work = pred_work.copy()
        true_work["formula_source_target"] = "delta_pka_pred"
        true_work["formula_source_subset"] = source_subset
        true_work["formula_source_model"] = source_model
        true_work["formula_source_fit_kind"] = "full_data"
        true_work["formula_source_selection_metric"] = str(used_metric or "")
        true_work["formula_training_observed_delta_pka_pred"] = pd.to_numeric(true_work["observed"], errors="coerce")
        true_work["experimental_observed_delta_pka_true"] = pd.to_numeric(true_work[exp_col], errors="coerce")
        true_work["observed"] = true_work["experimental_observed_delta_pka_true"]
        true_work["target"] = "delta_pka_true"
        true_work["applied_formula_note"] = (
            "same final delta_pka_pred formula applied to experimental delta_pka_true; "
            "no refit on true values"
        )
        true_work = true_work.dropna(subset=["observed", "fitted_full_data"]).copy()
        true_full_data = prepare_unique_full_data_plot_rows(
            plot_df=true_work,
            target="delta_pka_true",
            pred_col="fitted_full_data",
            keep_distinct_observed_values=True,
        )
        true_full_data = _order_full_data_columns(true_full_data, pred_col="fitted_full_data")
        true_full_data.to_csv(os.path.join(output, ACIDIC_TRUE_FULL_DATA_CSV), index=False)
        true_full_data.to_csv(os.path.join(output, TRUE_FULL_DATA_CSV), index=False)
        tm = metrics(true_full_data["observed"].values, true_full_data["fitted_full_data"].values) if len(true_full_data) else {"n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan}
        summary_rows.append({
            "table": "true_full_data",
            "csv": TRUE_FULL_DATA_CSV,
            "alias_csv": ACIDIC_TRUE_FULL_DATA_CSV,
            "evaluation_target": "delta_pka_true",
            "formula_source": "delta_pka_pred",
            "source_subset": source_subset,
            "source_model": source_model,
            "source_selection_metric": str(used_metric or ""),
            "final_formula_transform": FINAL_FORMULA_TRANSFORM,
            "dedup_rule": "same SMILES + same delta_pka",
            "n": int(tm["n"]),
            "r2": tm["r2"],
            "rmse": tm["rmse"],
            "mae": tm["mae"],
            "note": "acidic experimental true_full_data evaluated with final pred formula; no true-target refit",
        })
    else:
        empty_true = pd.DataFrame()
        empty_true.to_csv(os.path.join(output, ACIDIC_TRUE_FULL_DATA_CSV), index=False)
        empty_true.to_csv(os.path.join(output, TRUE_FULL_DATA_CSV), index=False)
        summary_rows.append({
            "table": "true_full_data",
            "csv": TRUE_FULL_DATA_CSV,
            "alias_csv": ACIDIC_TRUE_FULL_DATA_CSV,
            "evaluation_target": "delta_pka_true",
            "formula_source": "delta_pka_pred",
            "source_subset": source_subset,
            "source_model": source_model,
            "source_selection_metric": str(used_metric or ""),
            "final_formula_transform": FINAL_FORMULA_TRANSFORM,
            "dedup_rule": "same SMILES + same delta_pka",
            "n": 0,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "note": "no experimental delta_pka_true column found for true_full_data",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output, FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV), index=False)
    return summary_df


def export_transferred_pred_formula_to_true_outputs(
    pred_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output: str,
    args: argparse.Namespace,
    save_plot: bool = True,
) -> pd.DataFrame:


    w, source_best, used_metric, summary = build_pred_formula_applied_to_true_table(
        pred_df=pred_df,
        fit_df=fit_df,
        selection_metric=_best_parity_selection_metric(fit_df),
    )
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output, "transferred_pred_formula_to_true_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    if source_best is None or w.empty:
        return summary_df

    formula_prefix = "best_parity_delta_pka_true_full_data"
    selected_for_export = source_best.copy()
    selected_for_export["applied_to_target"] = "delta_pka_true"
    selected_for_export["formula_source_target"] = "delta_pka_pred"
    selected_for_export["selection_rule"] = (
        f"formula_from_delta_pka_pred_{used_metric}_applied_to_delta_pka_true_without_refit"
    )
    export_selected_formula_parameter_details(
        fit_df=fit_df,
        selected_row=selected_for_export,
        output_dir=output,
        args=args,
        file_prefix=formula_prefix,
        table_scope="delta_pka_pred_formula_applied_to_delta_pka_true",
    )

    dummy_figure_path = os.path.join(fig_dir(output), "best_parity_delta_pka_true_full_data.png")
    source_rows_csv = export_best_parity_source_rows(
        plot_df=w,
        output=output,
        target="delta_pka_true",
        fit_kind="full_data",
        subset=str(source_best.get("subset", "")),
        model=str(source_best.get("model", "")),
        pred_col="fitted_full_data",
        figure_path=dummy_figure_path,
    )
    summary_df["source_rows_csv"] = source_rows_csv
    summary_df["formula_summary_csv"] = f"{formula_prefix}_formula_summary.csv"
    summary_df["formula_parameters_csv"] = f"{formula_prefix}_formula_parameters.csv"
    summary_df["formula_report_md"] = f"{formula_prefix}_formula_with_parameters.md"
    summary_df.to_csv(summary_path, index=False)

    if not save_plot:
        return summary_df

    m = metrics(w["observed"].values, w["fitted_full_data"].values)
    lo, hi = parity_limits(w["observed"].values, w["fitted_full_data"].values)
    fig, ax = plt.subplots(figsize=(6.0, 5.3))
    for h in HALOGEN_ORDER:
        hh = w[w["halogen"] == h] if "halogen" in w.columns else pd.DataFrame()
        if len(hh):
            ax.scatter(hh["observed"], hh["fitted_full_data"], s=20, alpha=0.55, label=h)
    if "halogen" not in w.columns:
        ax.scatter(w["observed"], w["fitted_full_data"], s=20, alpha=0.55)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed delta_pka_true")
    ax.set_ylabel("HII-fitted delta_pka_true")
    ax.set_title(
        f"delta_pka_true: {source_best['model']} ({source_best['subset']}, full_data)\n"
        f"formula from delta_pka_pred; R²={m['r2']:.3f}, RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, n={m['n']}"
    )
    if "halogen" in w.columns:
        ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_file = os.path.join(fig_dir(output), "best_parity_delta_pka_true_full_data.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="acid_base_hii_fit_predictions.csv",
        source_columns=[
            "experimental_observed_delta_pka_true", "fitted_full_data", "halogen", "formula_source_target",
            "formula_source_model", "halogenated_original_smiles", "halogenated_original_pka",
            "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
        ],
        plot_type="best_parity_formula_transfer_to_experiment",
        target="delta_pka_true",
        subset=str(source_best.get("subset", "")),
        model=str(source_best.get("model", "")),
        fit_kind="full_data",
        exact_source_rows_csv=source_rows_csv,
        note=(
            "This full-data experimental parity plot reuses the best formula learned from delta_pka_pred. "
            "The HII formula is not refit on delta_pka_true; x is experimental delta_pka_true and y is the "
            "delta_pka_pred-formula fitted_full_data value."
        ),
    )
    plt.close(fig)
    return summary_df


def plot_delta_pka_true_full_data_using_pred_formula(
    pred_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output: str,
    args: argparse.Namespace,
    selection_metric: Optional[str] = None,
) -> bool:

    summary_df = export_transferred_pred_formula_to_true_outputs(
        pred_df=pred_df,
        fit_df=fit_df,
        output=output,
        args=args,
        save_plot=True,
    )
    return bool(summary_df is not None and not summary_df.empty and pd.notna(summary_df.iloc[0].get("r2", np.nan)))


def compute_transferred_pred_formula_to_true_metrics(
    pred_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    selection_metric: Optional[str] = None,
) -> Dict[str, Any]:


    rec: Dict[str, Any] = {
        "formula_source": "delta_pka_pred",
        "evaluation_target": "delta_pka_true",
        "fit_kind": "full_data",
        "selection_metric": selection_metric or "",
        "source_subset": "",
        "source_model": "",
        "n": 0,
        "r2": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "note": "delta_pka_pred formula applied to experimental delta_pka_true without refit",
    }
    if pred_df is None or pred_df.empty or fit_df is None or fit_df.empty:
        rec["note"] = "missing prediction or fit table"
        return rec
    if "target" not in pred_df.columns or "fitted_full_data" not in pred_df.columns:
        rec["note"] = "prediction table lacks target or fitted_full_data column"
        return rec

    source_best, used_metric = _select_best_parity_fit_row(
        fit_df=fit_df,
        target="delta_pka_pred",
        selection_metric=selection_metric,
    )
    rec["selection_metric"] = used_metric or ""
    if source_best is None:
        rec["note"] = "no fitted delta_pka_pred formula row found"
        return rec

    source_subset = str(source_best.get("subset", ""))
    source_model = str(source_best.get("model", ""))
    rec["source_subset"] = source_subset
    rec["source_model"] = source_model

    work = pred_df[
        (pred_df["target"].astype(str) == "delta_pka_pred")
        & (pred_df["subset"].astype(str) == source_subset)
        & (pred_df["model"].astype(str) == source_model)
    ].copy()
    if work.empty:
        rec["note"] = "no prediction rows found for selected delta_pka_pred formula"
        return rec

    exp_col = _find_experimental_delta_column(work)
    if exp_col is None:
        rec["note"] = "no experimental delta_pka_true column found on selected formula rows"
        return rec

    y_true = pd.to_numeric(work[exp_col], errors="coerce")
    y_hat = pd.to_numeric(work["fitted_full_data"], errors="coerce")
    valid = y_true.notna() & y_hat.notna()
    if int(valid.sum()) < 2:
        rec["note"] = "fewer than two matched experimental rows for transfer metric"
        return rec

    m = metrics(y_true[valid].values, y_hat[valid].values)
    rec.update({
        "n": int(m["n"]),
        "r2": float(m["r2"]),
        "rmse": float(m["rmse"]),
        "mae": float(m["mae"]),
        "experimental_column": exp_col,
        "note": "delta_pka_pred formula applied to experimental delta_pka_true without refit",
    })
    return rec


def print_transferred_pred_formula_to_true_summary(transfer_metrics: Dict[str, Any]) -> None:

    print("Transferred pred-formula to true:")
    print(f"  formula_source = {transfer_metrics.get('formula_source', 'delta_pka_pred')}")
    print(f"  evaluation_target = {transfer_metrics.get('evaluation_target', 'delta_pka_true')}")
    r2 = transfer_metrics.get("r2", np.nan)
    rmse = transfer_metrics.get("rmse", np.nan)
    mae = transfer_metrics.get("mae", np.nan)
    n = int(transfer_metrics.get("n", 0) or 0)
    if np.isfinite(r2):
        print(f"  R² = {float(r2):.4f}")
    else:
        print("  R² = NA")
    if np.isfinite(rmse):
        print(f"  RMSE = {float(rmse):.4f}")
    else:
        print("  RMSE = NA")
    if np.isfinite(mae):
        print(f"  MAE = {float(mae):.4f}")
    else:
        print("  MAE = NA")
    print(f"  n = {n}")
    if transfer_metrics.get("source_model") or transfer_metrics.get("source_subset"):
        print(
            f"  source_model = {transfer_metrics.get('source_model', '')} "
            f"({transfer_metrics.get('source_subset', '')})"
        )
    if transfer_metrics.get("selection_metric"):
        print(f"  source_selection_metric = {transfer_metrics.get('selection_metric')}")
    print(f"  note = {transfer_metrics.get('note', 'no refit on delta_pka_true')}")

def plot_best_parity(pred_df: pd.DataFrame, fit_df: pd.DataFrame, target: str, output: str, args: argparse.Namespace) -> None:
    if pred_df.empty:
        return

    selection_metric = _best_parity_selection_metric(fit_df)
    if selection_metric is None:
        return

    if "target" not in pred_df.columns:
        warnings.warn(
            "Prediction table has no 'target' column; cannot make target-specific parity plots. "
            "Regenerate acid_base_hii_fit_predictions.csv with this fixed script."
        )
        return

    best, used_metric = _select_best_parity_fit_row(
        fit_df=fit_df,
        target=target,
        selection_metric=selection_metric,
    )
    if best is None:
        return

    work = pred_df[
        (pred_df["target"].astype(str) == str(target))
        & (pred_df["subset"].astype(str) == str(best["subset"]))
        & (pred_df["model"].astype(str) == str(best["model"]))
    ].copy()
    if work.empty:
        return

    for col, label in FIT_PREDICTION_PLOT_SERIES:
        if col not in work.columns:
            continue
        w = work.dropna(subset=["observed", col]).copy()
        if label == "full_data":
            w = prepare_unique_full_data_plot_rows(
                plot_df=w,
                target=target,
                pred_col=col,
                keep_distinct_observed_values=True,
            )
        if len(w) < 3:
            continue
        m = metrics(w["observed"].values, w[col].values)
                                                                            
                                                                             
                                                                            
        metric_prefix = "full_data" if label == "full_data" else label
        display_r2 = best.get(f"{metric_prefix}_r2", m["r2"])
        display_rmse = best.get(f"{metric_prefix}_rmse", m["rmse"])
        display_mae = best.get(f"{metric_prefix}_mae", m["mae"])
        if pd.isna(display_r2):
            display_r2 = m["r2"]
        if pd.isna(display_rmse):
            display_rmse = m["rmse"]
        if pd.isna(display_mae):
            display_mae = m["mae"]
        lo, hi = parity_limits(w["observed"].values, w[col].values)
        fig, ax = plt.subplots(figsize=(6.0, 5.3))
        for h in HALOGEN_ORDER:
            hh = w[w["halogen"] == h]
            if len(hh):
                ax.scatter(hh["observed"], hh[col], s=20, alpha=0.55, label=h)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
        ax.axhline(0, linestyle=":", linewidth=1)
        ax.axvline(0, linestyle=":", linewidth=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed {target}")
        ax.set_ylabel(f"HII-fitted {target}")
        ax.set_title(
            f"{target}: {best['model']} ({best['subset']}, {label})\n"
            f"R²(fitted, unchanged)={display_r2:.3f}, "
            f"RMSE(fitted)={display_rmse:.3f}, MAE(fitted)={display_mae:.3f}, n={m['n']}"
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_file = os.path.join(fig_dir(output), f"best_parity_{target}_{label}.png")
        save_figure_with_svg(fig, out_file)

        source_rows_csv = ""
        if label == "full_data":
            source_rows_csv = export_best_parity_source_rows(
                plot_df=w,
                output=output,
                target=target,
                fit_kind=label,
                subset=str(best.get("subset", "")),
                model=str(best.get("model", "")),
                pred_col=col,
                figure_path=out_file,
            )
            formula_prefix = f"best_parity_{_safe_feature_token(str(target)).lower()}_{_safe_feature_token(str(label)).lower()}"
            export_selected_formula_parameter_details(
                fit_df=fit_df,
                selected_row=best,
                output_dir=output,
                args=args,
                file_prefix=formula_prefix,
                table_scope="best_parity_full_data_formula",
            )

        register_figure_source(
            output=output,
            figure_path=out_file,
            source_csv="acid_base_hii_fit_predictions.csv",
            source_columns=[
                "observed", col, "halogen", "target", "subset", "model",
                "halogenated_original_smiles", "halogenated_original_pka",
                "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
            ],
            plot_type="best_parity",
            target=target,
            subset=str(best.get("subset", "")),
            model=str(best.get("model", "")),
            fit_kind=label,
            exact_source_rows_csv=source_rows_csv,
            note=f"Model/subset selected by {used_metric}; full_data is the fit on all rows in the selected subset; train is kept as a backward-compatible alias; cv is cross-validated when available. The full_data row export includes raw input SMILES/pKa and pre-halogenation parent SMILES/pKa when available.",
        )
        plt.close(fig)


def plot_specific_parity(
    pred_df: pd.DataFrame,
    target: str,
    subset: str,
    model: str,
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
    title_prefix: str,
) -> None:


    if pred_df.empty or "target" not in pred_df.columns:
        return
    work = pred_df[
        (pred_df["target"].astype(str) == str(target))
        & (pred_df["subset"].astype(str) == str(subset))
        & (pred_df["model"].astype(str) == str(model))
    ].copy()
    if work.empty:
        return

    for col, label in FIT_PREDICTION_PLOT_SERIES:
        if col not in work.columns:
            continue
        w = work.dropna(subset=["observed", col]).copy()
        if len(w) < 3:
            continue
        m = metrics(w["observed"].values, w[col].values)
        lo, hi = parity_limits(w["observed"].values, w[col].values)
        fig, ax = plt.subplots(figsize=(6.0, 5.3))
        if "halogen" in w.columns:
            for h in HALOGEN_ORDER:
                hh = w[w["halogen"] == h]
                if len(hh):
                    ax.scatter(hh["observed"], hh[col], s=20, alpha=0.55, label=h)
        else:
            ax.scatter(w["observed"], w[col], s=20, alpha=0.55)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
        ax.axhline(0, linestyle=":", linewidth=1)
        ax.axvline(0, linestyle=":", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed {target}")
        ax.set_ylabel(f"HII-fitted {target}")
        ax.set_title(
            f"{title_prefix}: {target} ({label})\n"
            f"{model} / {subset}\n"
            f"R²={m['r2']:.3f}, RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, n={m['n']}"
        )
        if "halogen" in w.columns:
            ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_name = f"{filename_prefix}_{target}_{label}.png"
        out_file = os.path.join(fig_dir(output), out_name)
        save_figure_with_svg(fig, out_file)
        register_figure_source(
            output=output,
            figure_path=out_file,
            source_csv="acid_base_hii_fit_predictions.csv",
            source_columns=["observed", col, "halogen", "target", "subset", "model"],
            plot_type="specific_parity",
            target=target,
            subset=subset,
            model=model,
            fit_kind=label,
            note="full_data is the fitted-on-all-data result for the requested subset/model; cv is available only when cross-validation was run.",
        )
        plt.close(fig)


                                                          
                      
                                     
                                      
                                  
GNBU_CMAP = plt.cm.GnBu


def gnbu_hex(value: float) -> str:

    return to_hex(GNBU_CMAP(float(np.clip(value, 0.0, 1.0))))


HALOGEN_PLOT_COLORS = {
    "F": gnbu_hex(0.38),
    "Cl": gnbu_hex(0.58),
    "Br": gnbu_hex(0.78),
    "I": gnbu_hex(0.95),
}

                                                                          
DISTRIBUTION_PLOT_COLORS = {
    "F": gnbu_hex(0.38),
    "Cl": gnbu_hex(0.58),
    "Br": gnbu_hex(0.78),
    "I": gnbu_hex(0.95),
}

                                          
                                                           
ORIGINAL_IRAS_DISTRIBUTION_PLOT_COLORS = {
    "F": "#9E9E9E",
    "Cl": "#6FA0C8",
    "Br": "#E57373",
    "I": "#9ED0E6",
}

                                                                   
                                                               
                                        
                                         
                                                               
                                                                        
                                          
SQUARE_DISTRIBUTION_FIGSIZE = (6.8, 6.8)
AXIS_LABEL_FONT_FAMILY = "Arial"
AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 24
IRAS_YTICK_LABEL_FONT_SIZE = 20
TITLE_FONT_SIZE = 22
AXIS_SPINE_LINEWIDTH = 2.6
AXIS_TICK_LINEWIDTH = 2.4
AXIS_TICK_LENGTH = 7

POSITION_CATEGORY_ORDER = [
    "ortho",
    "meta",
    "para",
    "same_ring_other",
    "same_aromatic_system",
    "aliphatic",
    "unknown",
]

DELTA_PKA_AXIS_LABEL = r"$\Delta \mathrm{pK}_{\mathrm{a}}$"

HALOGEN_DISTRIBUTION_TICK_LABELS = {
    "F": "Fluoride",
    "Cl": "Chloride",
    "Br": "Bromide",
    "I": "Iodide",
}

POSITION_DISTRIBUTION_TICK_LABELS = {
    "ortho": "Ortho",
    "meta": "Meta",
    "para": "Para",
    "same_ring_other": "Same Ring Other",
    "same_aromatic_system": "Aromatic",
    "aliphatic": "Aliphatic",
    "unknown": "Other",
}

IRAS_COMPONENT_TICK_LABELS = {
    "I": "Inductive",
    "R": "Resonance",
    "A": "Aromatic",
    "S": "Steric",
    "Scaffold": "Scaffold",
}

                                   
                                                           
                                                                             
                                                                              
                             
                                 
                                      
HORIZONTAL_IRAS_SCAFFOLD_TERMS = [
    ("scaffold_TPSA", 0.0023094, "TPSA"),
    ("scaffold_MolLogP", 0.138223, "MolLogP"),
    ("scaffold_FractionCSP3", 0.126668, "FractionCSP3"),
]
HORIZONTAL_IRAS_SCAFFOLD_COMPONENT = "Scaffold"
HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL = "horizontal_iras_scaffold_delta_pka_contribution"
HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL = (
    "0.0023094*scaffold_TPSA + "
    "0.138223*scaffold_MolLogP + "
    "0.126668*scaffold_FractionCSP3"
)
HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA = (
    "0.0023094*TPSA + "
    "0.138223*MolLogP + "
    "0.126668*FractionCSP3"
)
HORIZONTAL_IRAS_SCAFFOLD_COLOR = "#CAE1FF"
                            

                                                                     
                                                                        
                                                                          
 
                                                      
                                                      
                                                     
                                                   
                                         
                                            
                                                 
 
                                                                           
                                                                          
HORIZONTAL_IRAS_FORMULA_TARGET = "delta_pka_pred"
HORIZONTAL_IRAS_FORMULA_INTERCEPT = 0.417006
HORIZONTAL_IRAS_FIXED_COMPONENT_TERMS = [
    ("acid_hii_inductive_selected", -4.01896, "I", "Inductive (I)"),
    ("acid_hii_resonance_selected", 0.925079, "R", "Resonance (R)"),
    ("acid_hii_aromatic_selected", -0.889082, "A", "Aromatic (A)"),
    ("acid_hii_steric_selected", -1.62732, "S", "Steric (S)"),
]
HORIZONTAL_IRAS_FIXED_COEFFICIENTS = {
    col: float(coef)
    for col, coef, _short_label, _title in HORIZONTAL_IRAS_FIXED_COMPONENT_TERMS
}
HORIZONTAL_IRAS_FORMULA_DISPLAY = (
    "delta_pka_pred = 0.417006 "
    "- 4.01896*I_acid "
    "+ 0.925079*R_acid "
    "- 0.889082*A_acid "
    "- 1.62732*S_acid "
    "+ 0.0023094*TPSA "
    "+ 0.138223*MolLogP "
    "+ 0.126668*FractionCSP3"
)


POSITION_PLOT_COLORS = {
    "ortho": gnbu_hex(0.30),
    "meta": gnbu_hex(0.42),
    "para": gnbu_hex(0.54),
    "same_ring_other": gnbu_hex(0.66),
    "same_aromatic_system": gnbu_hex(0.78),
    "aliphatic": gnbu_hex(0.88),
    "unknown": gnbu_hex(0.96),
}


def _safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def apply_requested_distribution_axis_format(
    ax,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotate_xticks: Optional[float] = None,
) -> None:

    ax.set_box_aspect(1)

    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            fontname=AXIS_LABEL_FONT_FAMILY,
            labelpad=8,
        )

    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            fontname=AXIS_LABEL_FONT_FAMILY,
            labelpad=8,
        )

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=AXIS_TICK_LINEWIDTH,
        length=AXIS_TICK_LENGTH,
        labelsize=TICK_LABEL_FONT_SIZE,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick.set_fontweight("bold")
        if rotate_xticks is not None:
            tick.set_rotation(rotate_xticks)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
        else:
            tick.set_rotation(0)
            tick.set_ha("center")
            tick.set_rotation_mode("default")

    for tick in ax.get_yticklabels():
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick.set_fontweight("bold")

    ax.title.set_fontname(AXIS_LABEL_FONT_FAMILY)
    ax.title.set_fontsize(TITLE_FONT_SIZE)
    ax.title.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)


def grouped_violin_box_scatter(
    ax,
    groups: List[np.ndarray],
    labels: List[str],
    ylabel: str,
    title: str,
    palette: Optional[List[str]] = None,
) -> None:

    valid = [
        (np.asarray(g, dtype=float), lab, i)
        for i, (g, lab) in enumerate(zip(groups, labels))
        if len(g) > 0
    ]
    if not valid:
        ax.set_axis_off()
        return

    groups = [g[np.isfinite(g)] for g, _, _ in valid]
    filtered = [(g, lab, i) for g, (_, lab, i) in zip(groups, valid) if len(g) > 0]
    if not filtered:
        ax.set_axis_off()
        return

    groups = [g for g, _, _ in filtered]
    labels = [lab for _, lab, _ in filtered]
    original_indices = [i for _, _, i in filtered]
    positions = np.arange(1, len(groups) + 1, dtype=float)

    if palette is not None:
        colors = [palette[i] if i < len(palette) else "#4C78A8" for i in original_indices]
    else:
        colors = [HALOGEN_PLOT_COLORS.get(lab, "#4C78A8") for lab in labels]

    violin_centers = positions + 0.14
    parts = ax.violinplot(
        groups,
        positions=violin_centers,
        widths=0.58,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color, center in zip(parts["bodies"], colors, violin_centers):
        path = body.get_paths()[0]
        verts = path.vertices
        verts[:, 0] = np.maximum(verts[:, 0], center)
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.22)
        body.set_zorder(1)

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor("#2A2A2A")
        patch.set_linewidth(1.6)
        patch.set_zorder(4)

    for key in ["whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#2A2A2A")
            item.set_linewidth(1.5)
            item.set_zorder(4)

    for item in bp["medians"]:
        item.set_color("#2A2A2A")
        item.set_linewidth(2.0)
        item.set_zorder(5)

    rng = np.random.default_rng(2026)
    for pos, vals, color in zip(positions, groups, colors):
        x = pos - 0.22 + rng.uniform(-0.075, 0.075, size=len(vals))
        ax.scatter(
            x,
            vals,
            s=78,
            alpha=0.80,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    all_vals = np.concatenate(groups)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals):
        ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
        pad = 0.12 * (ymax - ymin) if ymax > ymin else 0.5
        ax.set_ylim(ymin - pad, ymax + pad)

    category_side_pad = 0.46
    ax.set_xlim(
        float(positions[0] - category_side_pad),
        float(positions[-1] + category_side_pad),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        fontsize=TICK_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
        labelpad=8,
    )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
        pad=18,
    )

    ax.grid(False)
    ax.set_facecolor("white")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=AXIS_TICK_LINEWIDTH,
        length=AXIS_TICK_LENGTH,
        labelsize=TICK_LABEL_FONT_SIZE,
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)


def plot_halogen_delta_distribution_panel(
    site_df: pd.DataFrame,
    target: str,
    output: str,
    args: argparse.Namespace,
) -> None:
    if target not in site_df.columns or "halogen" not in site_df.columns:
        return

    work = site_df[[target, "halogen"]].copy()
    work[target] = _as_float_series(work[target])
    work["halogen"] = work["halogen"].astype(str)
    work = work.dropna(subset=[target])
    work = work[work["halogen"].isin(HALOGEN_ORDER)].copy()

    if len(work) < 3:
        return

    fig, ax = plt.subplots(figsize=SQUARE_DISTRIBUTION_FIGSIZE)

    groups = [
        work.loc[work["halogen"] == h, target].astype(float).values
        for h in HALOGEN_ORDER
    ]
    labels = [HALOGEN_DISTRIBUTION_TICK_LABELS.get(h, h) for h in HALOGEN_ORDER]
    colors = [
        DISTRIBUTION_PLOT_COLORS.get(h, HALOGEN_PLOT_COLORS[h])
        for h in HALOGEN_ORDER
    ]

    grouped_violin_box_scatter(
        ax,
        groups,
        labels,
        ylabel=DELTA_PKA_AXIS_LABEL,
        title=f"Halogen distribution of {target}",
        palette=colors,
    )

    apply_requested_distribution_axis_format(
        ax,
        ylabel=DELTA_PKA_AXIS_LABEL,
        rotate_xticks=30,
    )


    out_file = os.path.join(fig_dir(output), f"halogen_distribution_{target}.png")
    save_figure_with_svg(fig, out_file)

    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=[target, "halogen"],
        plot_type="halogen_distribution",
        target=target,
        note="Distribution of target values grouped by halogen identity.",
    )

    plt.close(fig)

def _normalize_position_category(value: Any) -> str:

    token = str(value).strip().lower() if value is not None else ""
    if token in {"", "nan", "none", "na", "null"}:
        return "unknown"
    token = token.replace("-", "_").replace(" ", "_")
    if token in POSITION_CATEGORY_ORDER:
        return token
                                                         
    alias = {
        "same_aromatic": "same_aromatic_system",
        "same_aromatic_ring_system": "same_aromatic_system",
        "same_ring": "same_ring_other",
        "other_same_ring": "same_ring_other",
        "aliph": "aliphatic",
    }.get(token)
    return alias if alias in POSITION_CATEGORY_ORDER else "unknown"


def _select_position_column(site_df: pd.DataFrame) -> Optional[str]:

    candidates = [
        "position",
        "target_position",
        "target_hii_position",
        "acid_position",
        "base_position",
        "site_position",
    ]
    for col in candidates:
        if col not in site_df.columns:
            continue
        vals = site_df[col].dropna().astype(str).str.strip().str.lower()
        if vals.empty:
            continue
        normalized = vals.map(_normalize_position_category)
        if normalized.isin(POSITION_CATEGORY_ORDER).any():
            return col
    return None


def plot_position_delta_distribution_panel(
    site_df: pd.DataFrame,
    target: str,
    output: str,
    args: argparse.Namespace,
) -> None:

    if target not in site_df.columns:
        return

    pos_col = _select_position_column(site_df)
    if pos_col is None:
        return

    work = site_df[[target, pos_col]].copy()
    work[target] = _as_float_series(work[target])
    work["position_category"] = work[pos_col].map(_normalize_position_category)
    work = work.dropna(subset=[target]).copy()
    work = work[work["position_category"].isin(POSITION_CATEGORY_ORDER)].copy()

    if len(work) < 3:
        return

    groups = [
        work.loc[work["position_category"] == cat, target].astype(float).values
        for cat in POSITION_CATEGORY_ORDER
    ]
    labels = [
        POSITION_DISTRIBUTION_TICK_LABELS.get(cat, cat)
        for cat in POSITION_CATEGORY_ORDER
    ]
    colors = [
        POSITION_PLOT_COLORS.get(cat, "#4C78A8")
        for cat in POSITION_CATEGORY_ORDER
    ]

    if sum(len(g) for g in groups) < 3:
        return

    fig, ax = plt.subplots(figsize=SQUARE_DISTRIBUTION_FIGSIZE)

    grouped_violin_box_scatter(
        ax,
        groups,
        labels,
        ylabel=DELTA_PKA_AXIS_LABEL,
        title=f"Position distribution of {target}",
        palette=colors,
    )

    apply_requested_distribution_axis_format(
        ax,
        ylabel=DELTA_PKA_AXIS_LABEL,
        rotate_xticks=30,
    )

    fig.tight_layout()

    out_file = os.path.join(fig_dir(output), f"position_distribution_{target}.png")
    save_figure_with_svg(fig, out_file)

    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=[target, pos_col],
        plot_type="position_distribution",
        target=target,
        note=(
            "Distribution of target values grouped by relative halogen/center position. "
            "Empty position categories are dropped before tick labels are styled."
        ),
    )

    plt.close(fig)


def plot_formula_distribution_panels(
    site_df: pd.DataFrame,
    target: str,
    subset_name: str,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    output: str,
    args: argparse.Namespace,
) -> None:


    if target not in site_df.columns:
        return
    if any(c not in site_df.columns for c in feature_cols):
        return

    work = site_df.loc[subset_mask].copy()
    if len(work) < 5:
        return

                                                                             
                                                                                 
    work[target] = _as_float_series(work[target])
    for col in feature_cols:
        work[col] = _as_float_series(work[col])
    work = work.dropna(subset=[target] + feature_cols).copy()
    if len(work) < 5:
        return

    component_matrix = work[feature_cols].astype(float).to_numpy()
    if component_matrix.size == 0:
        return

    dominant_idx = np.nanargmax(np.abs(component_matrix), axis=1)
    work["_dominant_formula_component"] = dominant_idx

    groups: List[np.ndarray] = []
    labels: List[str] = []
    for idx, title in enumerate(feature_titles):
        vals = work.loc[work["_dominant_formula_component"] == idx, target].astype(float).values
        groups.append(vals if len(vals) else np.asarray([], dtype=float))
        if "(" in title and ")" in title:
            labels.append(title.split("(")[-1].split(")")[0])
        else:
            labels.append(title)

    if sum(len(g) for g in groups) < 3:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    base_colors = [
        DISTRIBUTION_PLOT_COLORS["F"],
        DISTRIBUTION_PLOT_COLORS["Cl"],
        DISTRIBUTION_PLOT_COLORS["Br"],
        DISTRIBUTION_PLOT_COLORS["I"],
    ]
                                                               
    colors = [base_colors[i % len(base_colors)] for i in range(len(feature_cols))]
    grouped_violin_box_scatter(
        ax,
        groups,
        labels,
        ylabel=target,
        title=f"{subset_name}\nΔpKa distribution by dominant formula component ({target})",
        palette=colors,
    )
    fig.tight_layout()
    safe_name = subset_name.lower().replace(' ', '_').replace('/', '_')
    out_file = os.path.join(fig_dir(output), f'{safe_name}_formula_distribution_{target}.png')
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=[target] + feature_cols,
        plot_type="formula_distribution_by_dominant_component",
        target=target,
        subset=subset_name,
        note="Y-axis is the target value; component columns are used to assign each row to its dominant formula component.",
    )
    plt.close(fig)


def grouped_horizontal_violin_box_scatter(
    ax,
    groups: List[np.ndarray],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    palette: Optional[List[str]] = None,
) -> None:

    valid = [
        (np.asarray(g, dtype=float), lab, i)
        for i, (g, lab) in enumerate(zip(groups, labels))
        if len(g) > 0
    ]
    if not valid:
        ax.set_axis_off()
        return

    groups = [g[np.isfinite(g)] for g, _, _ in valid]
    filtered = [(g, lab, i) for g, (_, lab, i) in zip(groups, valid) if len(g) > 0]
    if not filtered:
        ax.set_axis_off()
        return

    groups = [g for g, _, _ in filtered]
    labels = [lab for _, lab, _ in filtered]
    original_indices = [i for _, _, i in filtered]
    positions = np.arange(1, len(groups) + 1, dtype=float)

    if palette is not None:
        colors = [palette[i] if i < len(palette) else "#4C78A8" for i in original_indices]
    else:
        colors = [HALOGEN_PLOT_COLORS.get(lab, "#4C78A8") for lab in labels]

    violin_centers = positions + 0.22
    parts = ax.violinplot(
        groups,
        positions=violin_centers,
        widths=0.55,
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color, center in zip(parts["bodies"], colors, violin_centers):
        path = body.get_paths()[0]
        verts = path.vertices
        verts[:, 1] = np.maximum(verts[:, 1], center)
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_linewidth(1.5)
        body.set_alpha(0.55)
        body.set_zorder(1)

    box_centers = positions - 0.23
    bp = ax.boxplot(
        groups,
        positions=box_centers,
        vert=False,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.92)
        patch.set_edgecolor("#2A2A2A")
        patch.set_linewidth(2.0)
        patch.set_zorder(4)

    for key in ["whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#2A2A2A")
            item.set_linewidth(1.7)
            item.set_zorder(4)

    for item in bp["medians"]:
        item.set_color("#2A2A2A")
        item.set_linewidth(2.4)
        item.set_zorder(5)

    rng = np.random.default_rng(2026)
    for pos, vals, color in zip(positions, groups, colors):
        y = pos + rng.uniform(-0.075, 0.075, size=len(vals))
        ax.scatter(
            vals,
            y,
            s=76,
            alpha=0.78,
            color=color,
            edgecolors="#2A2A2A",
            linewidths=0.8,
            zorder=3,
        )

    all_vals = np.concatenate(groups)
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals):
        xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
        pad = 0.12 * (xmax - xmin) if xmax > xmin else 0.5
        ax.set_xlim(xmin - pad, xmax + pad)

    ax.set_ylim(0.45, len(groups) + 0.85)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        labels,
        fontsize=TICK_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
        labelpad=8,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
        labelpad=8,
    )
    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=AXIS_LABEL_FONT_FAMILY,
        pad=16,
    )

    ax.grid(False)
    ax.set_facecolor("white")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=AXIS_TICK_LINEWIDTH,
        length=AXIS_TICK_LENGTH,
        labelsize=TICK_LABEL_FONT_SIZE,
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)


def _select_best_acid_formula_row_for_iras_plot(
    fit_df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
) -> Tuple[Optional[pd.Series], Dict[str, float], Optional[str]]:


    coef_map = {
        col: float(HORIZONTAL_IRAS_FIXED_COEFFICIENTS.get(col, np.nan))
        for col in feature_cols
    }
    if any(not np.isfinite(v) for v in coef_map.values()):
        return None, {}, None

    if fit_df is None or fit_df.empty or "target" not in fit_df.columns:
        return None, coef_map, None

    selection_metric = _best_parity_selection_metric(fit_df)
    if selection_metric is None or selection_metric not in fit_df.columns:
        return None, coef_map, selection_metric

    work = fit_df[(fit_df["target"].astype(str) == str(target)) & fit_df[selection_metric].notna()].copy()
    if work.empty:
        return None, coef_map, selection_metric

    best_metric = work[selection_metric].max()
    candidates = work[work[selection_metric] >= best_metric - 0.01].copy()

    def _prio(model_name: Any) -> int:
        model_name = str(model_name)
        if is_scaffold_triple_model(model_name):
            return 0
        priority = {
            "acid_fixed_components": 1,
            "acid_directed_components": 2,
            "target_hii_components": 3,
            "target_components_plus_pka_mode_interactions": 4,
            "target_hii_total": 5,
        }
        return int(priority.get(model_name, 9))

    candidates["_iras_plot_prio"] = candidates["model"].map(_prio)
    best = candidates.sort_values(["_iras_plot_prio", selection_metric, "n"], ascending=[True, False, False]).iloc[0]
    return best, coef_map, selection_metric


def horizontal_iras_scaffold_cols() -> List[str]:

    return [col for col, _, _ in HORIZONTAL_IRAS_SCAFFOLD_TERMS]


def has_horizontal_iras_scaffold_terms(df: pd.DataFrame) -> bool:

    return df is not None and all(col in df.columns for col in horizontal_iras_scaffold_cols())


def compute_horizontal_iras_scaffold_contribution(df: pd.DataFrame) -> pd.Series:


    if df is None or df.empty:
        return pd.Series(dtype=float)
    if not has_horizontal_iras_scaffold_terms(df):
        return pd.Series(np.nan, index=df.index, dtype=float)

    total = pd.Series(0.0, index=df.index, dtype=float)
    for col, coef, _label in HORIZONTAL_IRAS_SCAFFOLD_TERMS:
        total = total + float(coef) * pd.to_numeric(df[col], errors="coerce")
    return total.replace([np.inf, -np.inf], np.nan)


def add_horizontal_iras_scaffold_contribution_columns(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return df
    out = df.copy()
    out[HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL] = compute_horizontal_iras_scaffold_contribution(out)
    out["scaffold_component_formula"] = HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA
    out["scaffold_component_feature_col"] = HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL
    for col, coef, label in HORIZONTAL_IRAS_SCAFFOLD_TERMS:
        raw = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
        safe_label = _safe_feature_token(label)
        out[f"scaffold_component_{safe_label}_feature_col"] = col
        out[f"scaffold_component_{safe_label}_raw_value"] = raw
        out[f"scaffold_component_{safe_label}_coefficient"] = float(coef)
        out[f"scaffold_component_{safe_label}_delta_pka_contribution"] = raw * float(coef)
    return out


def _scaffold_long_component_record(base: Dict[str, Any]) -> Dict[str, Any]:

    contribution = _safe_float(base.get(HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL, np.nan))
    contribution_value = contribution if contribution is not None else np.nan
    rec = dict(base)
    rec.update({
        "component": HORIZONTAL_IRAS_SCAFFOLD_COMPONENT,
        "component_title": "Scaffold (TPSA + MolLogP + FractionCSP3)",
        "component_feature_col": HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL,
                                                                             
                                                                          
                                                                            
                                                       
        "component_raw_value": contribution_value,
        "component_raw_value_kind": "coefficient_weighted_scaffold_sum",
        "formula_coefficient": 1.0,
        "formula_coefficient_kind": "merged_fixed_scaffold_expression",
        "component_delta_pka_contribution": contribution_value,
        "component_abs_delta_pka_contribution": abs(contribution_value) if contribution is not None else np.nan,
        "scaffold_component_formula": HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA,
        "scaffold_component_feature_col": HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL,
    })
    for col, coef, label in HORIZONTAL_IRAS_SCAFFOLD_TERMS:
        raw = _safe_float(base.get(col, np.nan))
        safe_label = _safe_feature_token(label)
        rec[f"scaffold_component_{safe_label}_feature_col"] = col
        rec[f"scaffold_component_{safe_label}_raw_value"] = raw if raw is not None else np.nan
        rec[f"scaffold_component_{safe_label}_coefficient"] = float(coef)
        rec[f"scaffold_component_{safe_label}_delta_pka_contribution"] = (
            raw * float(coef) if raw is not None else np.nan
        )
    return rec


def _horizontal_iras_source_preferred_columns(
    feature_cols: List[str],
    target: str,
) -> List[str]:

    return [
        "sample_index_in_iras_source",
        "target_delta_pka_column",
        "target_delta_pka",
        "horizontal_iras_formula_target",
        "horizontal_iras_formula_intercept",
        "horizontal_iras_formula_display",
        "horizontal_iras_formula_note",
        "component",
        "component_title",
        "component_feature_col",
        "component_raw_value",
        "component_raw_value_kind",
        "formula_coefficient",
        "component_delta_pka_contribution",
        "component_abs_delta_pka_contribution",
        "scaffold_component_formula",
        "scaffold_component_feature_col",
        "scaffold_component_MolLogP_feature_col",
        "scaffold_component_MolLogP_raw_value",
        "scaffold_component_MolLogP_coefficient",
        "scaffold_component_MolLogP_delta_pka_contribution",
        "scaffold_component_TPSA_feature_col",
        "scaffold_component_TPSA_raw_value",
        "scaffold_component_TPSA_coefficient",
        "scaffold_component_TPSA_delta_pka_contribution",
        "scaffold_component_FractionCSP3_feature_col",
        "scaffold_component_FractionCSP3_raw_value",
        "scaffold_component_FractionCSP3_coefficient",
        "scaffold_component_FractionCSP3_delta_pka_contribution",
        "selected_formula_model",
        "selected_formula_subset",
        "selected_formula_target",
        "selected_formula_metric_name",
        "selected_formula_metric_value",
        "selected_formula_n",
        "selected_formula_n_features",
        "selected_formula_formula",
        "subset_name",
        "source_target_column",
        target,
        "site_id",
        "source_row",
        "halogen",
        "halogen_idx",
        "attached_idx",
        "attached_symbol",
        "attached_is_aromatic",
        "halogenated_smiles",
        "masked_parent_smiles",
        "raw_input_csv_path",
        "input_smiles_col",
        "input_pka_col",
        "halogenated_original_smiles",
        "halogenated_original_pka",
        "halogenated_original_pka_type",
        "dehalogenated_parent_original_smiles",
        "dehalogenated_parent_original_pka",
        "pre_halogenation_original_smiles",
        "pre_halogenation_original_pka",
        "pka_pred_halogenated",
        "pka_pred_parent_masked",
        "pka_true_halogenated",
        "pka_true_parent_matched",
        "delta_pka_pred",
        "delta_pka_true",
        "delta_error",
        "pka_mode_inferred",
        "target_hii_mode",
        "acid_type",
        "base_type",
        "target_center_type",
        "position",
        "acid_position",
        "base_position",
        "target_position",
        "ring_type",
        "acid_ring_type",
        "base_ring_type",
        "target_hii_total",
        "acid_hii_total_mean",
        "base_hii_total_mean",
    ] + list(feature_cols) + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS


def export_horizontal_iras_distribution_source_data(
    work: pd.DataFrame,
    fit_row: pd.Series,
    coef_map: Dict[str, float],
    target: str,
    subset_name: str,
    feature_cols: List[str],
    feature_titles: List[str],
    output: str,
    filename_prefix: str,
    selection_metric: Optional[str],
) -> Tuple[str, str]:


    if work is None or work.empty:
        return "", ""

    out_dir = output
    os.makedirs(out_dir, exist_ok=True)
    target_token = _safe_target_filename(target)
    long_file = f"{filename_prefix}_horizontal_iras_distribution_{target_token}_source_data.csv"
    wide_file = f"{filename_prefix}_horizontal_iras_distribution_{target_token}_source_data_wide.csv"

    source = work.copy().reset_index(drop=True)
    source = add_horizontal_iras_scaffold_contribution_columns(source)
    source.insert(0, "sample_index_in_iras_source", np.arange(len(source), dtype=int))
    source["target_delta_pka_column"] = str(target)
    source["target_delta_pka"] = pd.to_numeric(source[target], errors="coerce") if target in source.columns else np.nan
    source["subset_name"] = str(subset_name)
    source["source_target_column"] = str(target)
    source["horizontal_iras_formula_target"] = HORIZONTAL_IRAS_FORMULA_TARGET
    source["horizontal_iras_formula_intercept"] = float(HORIZONTAL_IRAS_FORMULA_INTERCEPT)
    source["horizontal_iras_formula_display"] = HORIZONTAL_IRAS_FORMULA_DISPLAY
    source["horizontal_iras_formula_note"] = (
        "I/R/A/S and Scaffold are plotted as coefficient-weighted formula-term "
        "contributions from the fixed fitted delta_pka_pred formula supplied by the user."
    )
    source["selected_formula_model"] = str(fit_row.get("model", ""))
    source["selected_formula_subset"] = str(fit_row.get("subset", ""))
    source["selected_formula_target"] = str(fit_row.get("target", target))
    source["selected_formula_metric_name"] = str(selection_metric or "")
    source["selected_formula_metric_value"] = fit_row.get(selection_metric, np.nan) if selection_metric else np.nan
    source["selected_formula_n"] = fit_row.get("n", np.nan)
    source["selected_formula_n_features"] = fit_row.get("n_features", np.nan)
    source["selected_formula_formula"] = fit_row.get("formula", "")

    long_rows: List[Dict[str, Any]] = []
    for _, row in source.iterrows():
        base = row.to_dict()
        for col, title in zip(feature_cols, feature_titles):
            short_label = _component_short_label(title)
            coef = float(coef_map.get(col, np.nan))
            raw_value = _safe_float(row.get(col, np.nan))
            contribution = np.nan if raw_value is None or not np.isfinite(coef) else float(raw_value * coef)
            rec = dict(base)
            rec.update({
                "component": short_label,
                "component_title": str(title),
                "component_feature_col": str(col),
                "component_raw_value": raw_value if raw_value is not None else np.nan,
                "component_raw_value_kind": "raw_hii_component_value",
                "formula_coefficient": coef,
                "formula_coefficient_kind": "fixed_requested_formula_coefficient",
                "component_delta_pka_contribution": contribution,
                "component_abs_delta_pka_contribution": abs(contribution) if np.isfinite(contribution) else np.nan,
            })
            long_rows.append(rec)

        scaffold_rec = _scaffold_long_component_record(base)
        if pd.notna(scaffold_rec.get("component_delta_pka_contribution", np.nan)):
            long_rows.append(scaffold_rec)

    long_df = pd.DataFrame(long_rows)
    preferred_long = _horizontal_iras_source_preferred_columns(feature_cols, target)
    ordered_long = [c for c in preferred_long if c in long_df.columns]
    ordered_long += [c for c in long_df.columns if c not in ordered_long]
    ordered_long = list(dict.fromkeys(ordered_long))
    long_df = long_df[ordered_long]
    long_df.to_csv(os.path.join(out_dir, long_file), index=False)

    wide_df = source.copy()
    for col, title in zip(feature_cols, feature_titles):
        short_label = _component_short_label(title)
        coef = float(coef_map.get(col, np.nan))
        raw = pd.to_numeric(wide_df[col], errors="coerce")
        prefix = str(short_label).replace(" ", "_")
        wide_df[f"{prefix}_feature_col"] = col
        wide_df[f"{prefix}_raw_value"] = raw
        wide_df[f"{prefix}_formula_coefficient"] = coef
        wide_df[f"{prefix}_delta_pka_contribution"] = raw * coef
        wide_df[f"{prefix}_abs_delta_pka_contribution"] = (raw * coef).abs()

    wide_df["Scaffold_feature_col"] = HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL
    wide_df["Scaffold_formula"] = HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA
    wide_df["Scaffold_formula_coefficient"] = 1.0
    wide_df["Scaffold_delta_pka_contribution"] = pd.to_numeric(
        wide_df[HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL], errors="coerce"
    )
    wide_df["Scaffold_abs_delta_pka_contribution"] = wide_df["Scaffold_delta_pka_contribution"].abs()
    for col, coef, label in HORIZONTAL_IRAS_SCAFFOLD_TERMS:
        raw = pd.to_numeric(wide_df[col], errors="coerce") if col in wide_df.columns else pd.Series(np.nan, index=wide_df.index)
        safe_label = _safe_feature_token(label)
        wide_df[f"Scaffold_{safe_label}_feature_col"] = col
        wide_df[f"Scaffold_{safe_label}_raw_value"] = raw
        wide_df[f"Scaffold_{safe_label}_coefficient"] = float(coef)
        wide_df[f"Scaffold_{safe_label}_delta_pka_contribution"] = raw * float(coef)

    preferred_wide = [
        "sample_index_in_iras_source", "target_delta_pka_column", "target_delta_pka",
        "horizontal_iras_formula_target", "horizontal_iras_formula_intercept",
        "horizontal_iras_formula_display", "horizontal_iras_formula_note",
        "selected_formula_model", "selected_formula_subset", "selected_formula_target",
        "selected_formula_metric_name", "selected_formula_metric_value",
        "selected_formula_formula", "subset_name", "source_target_column", target,
        "site_id", "source_row", "halogen", "halogen_idx", "attached_idx",
        "halogenated_smiles", "masked_parent_smiles", "raw_input_csv_path",
        "halogenated_original_smiles", "halogenated_original_pka",
        "dehalogenated_parent_original_smiles", "dehalogenated_parent_original_pka",
        "pre_halogenation_original_smiles", "pre_halogenation_original_pka",
        "delta_pka_pred", "delta_pka_true", "delta_error",
        "pka_mode_inferred", "target_hii_mode", "acid_type", "target_center_type",
        "Scaffold_feature_col", "Scaffold_formula", "Scaffold_formula_coefficient",
        "Scaffold_delta_pka_contribution", "Scaffold_abs_delta_pka_contribution",
        "Scaffold_MolLogP_feature_col", "Scaffold_MolLogP_raw_value",
        "Scaffold_MolLogP_coefficient", "Scaffold_MolLogP_delta_pka_contribution",
        "Scaffold_TPSA_feature_col", "Scaffold_TPSA_raw_value",
        "Scaffold_TPSA_coefficient", "Scaffold_TPSA_delta_pka_contribution",
        "Scaffold_FractionCSP3_feature_col", "Scaffold_FractionCSP3_raw_value",
        "Scaffold_FractionCSP3_coefficient", "Scaffold_FractionCSP3_delta_pka_contribution",
    ]
    for title in feature_titles:
        prefix = _component_short_label(title).replace(" ", "_")
        preferred_wide.extend([
            f"{prefix}_feature_col",
            f"{prefix}_raw_value",
            f"{prefix}_formula_coefficient",
            f"{prefix}_delta_pka_contribution",
            f"{prefix}_abs_delta_pka_contribution",
        ])
    preferred_wide += list(feature_cols) + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS
    ordered_wide = [c for c in preferred_wide if c in wide_df.columns]
    ordered_wide += [c for c in wide_df.columns if c not in ordered_wide]
    ordered_wide = list(dict.fromkeys(ordered_wide))
    wide_df = wide_df[ordered_wide]
    wide_df.to_csv(os.path.join(out_dir, wide_file), index=False)

    return long_file, wide_file


def plot_iras_horizontal_delta_distribution_panel(
    site_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    target: str,
    subset_name: str,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:

    if fit_df is None or fit_df.empty:
        return
    if target not in site_df.columns:
        return
    if any(c not in site_df.columns for c in feature_cols):
        return

    scaffold_cols = horizontal_iras_scaffold_cols()
    scaffold_available = has_horizontal_iras_scaffold_terms(site_df)

    keep_cols = []
    preferred_meta_cols = [
        "site_id", "source_row", "halogen", "halogen_idx", "attached_idx",
        "attached_symbol", "attached_is_aromatic", "halogenated_smiles",
        "masked_parent_smiles", "raw_input_csv_path", "input_smiles_col", "input_pka_col",
        "halogenated_original_smiles", "halogenated_original_pka",
        "halogenated_original_pka_type", "dehalogenated_parent_original_smiles",
        "dehalogenated_parent_original_pka", "pre_halogenation_original_smiles",
        "pre_halogenation_original_pka", "pka_pred_halogenated",
        "pka_pred_parent_masked", "pka_true_halogenated", "pka_true_parent_matched",
        "delta_pka_pred", "delta_pka_true", "delta_error", "pka_mode_inferred",
        "target_hii_mode", "acid_type", "base_type", "target_center_type",
        "position", "acid_position", "base_position", "target_position",
        "ring_type", "acid_ring_type", "base_ring_type", "target_hii_total",
        "acid_hii_total_mean", "base_hii_total_mean",
    ] + list(feature_cols) + SCAFFOLD_COMPONENT_COLS + SCAFFOLD_DESCRIPTOR_COLS

    for c in [target] + preferred_meta_cols:
        if c in site_df.columns and c not in keep_cols:
            keep_cols.append(c)

    work = site_df.loc[subset_mask, keep_cols].copy()
    if len(work) < 5:
        return

    work[target] = _as_float_series(work[target])
    for col in feature_cols:
        work[col] = _as_float_series(work[col])
    if scaffold_available:
        for col in scaffold_cols:
            work[col] = _as_float_series(work[col])

    required_plot_cols = [target] + feature_cols + (scaffold_cols if scaffold_available else [])
    work = work.dropna(subset=required_plot_cols).copy()
    if len(work) < 5:
        return
    if scaffold_available:
        work = add_horizontal_iras_scaffold_contribution_columns(work)

    fit_row, coef_map, selection_metric = _select_best_acid_formula_row_for_iras_plot(
        fit_df=fit_df,
        target=target,
        feature_cols=feature_cols,
    )
    if fit_row is None or not coef_map:
        return

    source_long_csv, source_wide_csv = export_horizontal_iras_distribution_source_data(
        work=work,
        fit_row=fit_row,
        coef_map=coef_map,
        target=target,
        subset_name=subset_name,
        feature_cols=feature_cols,
        feature_titles=feature_titles,
        output=output,
        filename_prefix=filename_prefix,
        selection_metric=selection_metric,
    )

    labels: List[str] = []
    groups: List[np.ndarray] = []
    coeff_text_parts: List[str] = []

    for col, title in zip(feature_cols, feature_titles):
        short_label = title.split("(")[-1].split(")")[0] if "(" in title and ")" in title else title
        coef = float(coef_map.get(col, np.nan))
        weighted_vals = work[col].astype(float).values * coef
        weighted_vals = weighted_vals[np.isfinite(weighted_vals)]
        labels.append(IRAS_COMPONENT_TICK_LABELS.get(short_label, short_label))
        groups.append(weighted_vals)
        coeff_text_parts.append(f"{short_label}={coef:.4f}")

    if scaffold_available and HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL in work.columns:
        scaffold_vals = pd.to_numeric(work[HORIZONTAL_IRAS_SCAFFOLD_CONTRIBUTION_COL], errors="coerce").to_numpy(dtype=float)
        scaffold_vals = scaffold_vals[np.isfinite(scaffold_vals)]
        if len(scaffold_vals) > 0:
            labels.append(IRAS_COMPONENT_TICK_LABELS.get(HORIZONTAL_IRAS_SCAFFOLD_COMPONENT, HORIZONTAL_IRAS_SCAFFOLD_COMPONENT))
            groups.append(scaffold_vals)
            coeff_text_parts.append(f"Scaffold={HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA}")

    if sum(len(g) for g in groups) < 3:
        return

    base_iras_colors = [
        ORIGINAL_IRAS_DISTRIBUTION_PLOT_COLORS["F"],
        ORIGINAL_IRAS_DISTRIBUTION_PLOT_COLORS["Cl"],
        ORIGINAL_IRAS_DISTRIBUTION_PLOT_COLORS["Br"],
        ORIGINAL_IRAS_DISTRIBUTION_PLOT_COLORS["I"],
    ]
    colors = base_iras_colors[:min(len(groups), len(base_iras_colors))]
    if len(groups) > len(colors):
        colors.extend([HORIZONTAL_IRAS_SCAFFOLD_COLOR] * (len(groups) - len(colors)))

    fig, ax = plt.subplots(figsize=SQUARE_DISTRIBUTION_FIGSIZE)

    grouped_horizontal_violin_box_scatter(
        ax,
        groups=groups,
        labels=labels,
        xlabel=DELTA_PKA_AXIS_LABEL,
        ylabel="IRAS / scaffold component",
        title=f"{subset_name}\nHorizontal fitted-formula contribution distribution ({target})",
        palette=colors,
    )
    
    apply_requested_distribution_axis_format(
        ax,
        xlabel=DELTA_PKA_AXIS_LABEL,
        ylabel="IRAS/scaffold component",
                                            
    )
                                                                                
    ax.yaxis.label.set_size(24)            
    
    for tick in ax.get_yticklabels():
        tick.set_fontsize(20)                
        tick.set_fontweight("bold")
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
                                             
                                                               
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    
                                                                                             
                                                                               
                                                                               
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
        tick.set_rotation_mode("default")
        tick.set_fontname(AXIS_LABEL_FONT_FAMILY)
        tick.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick.set_fontweight("bold")
    
    coeff_text = "  |  ".join(coeff_text_parts)
    metric_name = selection_metric or "full_data_r2"
    metric_val = fit_row.get(metric_name, np.nan)
    model_name = str(fit_row.get("model", ""))

    try:
        metric_text = f"{metric_name}={float(metric_val):.4f}"
    except Exception:
        metric_text = f"{metric_name}=NA"

    subtitle = f"Formula terms from user-fitted equation; selected run row: {model_name} ; {metric_text}"
    fig.text(
        0.5,
        0.012,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontname=AXIS_LABEL_FONT_FAMILY,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_file = os.path.join(
        fig_dir(output),
        f"{filename_prefix}_horizontal_iras_distribution_{target}.png",
    )
    save_figure_with_svg(fig, out_file)

    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv=["counterfactual_mask_acid_base_hii_sites.csv", "acid_base_hii_fit_summary.csv"],
        source_columns=(
            [target] + feature_cols + horizontal_iras_scaffold_cols()
            + ["horizontal_iras_formula_display", "horizontal_iras_formula_intercept"]
            + [HORIZONTAL_IRAS_SCAFFOLD_FEATURE_COL]
        ),
        plot_type="horizontal_iras_weighted_distribution",
        target=target,
        subset=subset_name,
        exact_source_rows_csv=source_long_csv,
        note=(
            "Rows are I/R/A/S plus one merged Scaffold row. All plotted x-values are "
            "coefficient-weighted formula-term contributions from the fixed user-supplied "
            "delta_pka_pred formula: I=-4.01896*I_acid, R=0.925079*R_acid, "
            "A=-0.889082*A_acid, S=-1.62732*S_acid, and Scaffold="
            f"{HORIZONTAL_IRAS_SCAFFOLD_DISPLAY_FORMULA}. Source data are exported to "
            f"{source_long_csv}; wide per-sample data are exported to {source_wide_csv}."
        ),
    )

    plt.close(fig)

def plot_formula_parameter_range_distribution_panel(
    site_df: pd.DataFrame,
    subset_name: str,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:


    if any(c not in site_df.columns for c in feature_cols):
        return
    work = site_df.loc[subset_mask, feature_cols].copy()
    if len(work) < 5:
        return
    for col in feature_cols:
        work[col] = _as_float_series(work[col])
    work = work.dropna(subset=feature_cols).copy()
    if len(work) < 5:
        return

    groups: List[np.ndarray] = [work[col].astype(float).values for col in feature_cols]
    labels: List[str] = []
    for title in feature_titles:
        labels.append(_component_short_label(title) if "(" in title and ")" in title else title)
    if sum(len(g) for g in groups) < 3:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    base_colors = [
        DISTRIBUTION_PLOT_COLORS["F"],
        DISTRIBUTION_PLOT_COLORS["Cl"],
        DISTRIBUTION_PLOT_COLORS["Br"],
        DISTRIBUTION_PLOT_COLORS["I"],
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(feature_cols))]
    grouped_violin_box_scatter(
        ax,
        groups,
        labels,
        ylabel="HII component value",
        title=f"{subset_name}\nRange distribution of four acid HII parameters",
        palette=colors,
    )
    fig.tight_layout()
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}_parameter_range_distribution.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=feature_cols,
        plot_type="formula_parameter_range_distribution",
        subset=subset_name,
        note="Y-axis is the actual HII component value range for the selected formula components.",
    )
    plt.close(fig)


COMPONENT_PROJECTION_COLORS = {
    "Inductive (I)": "#4C78A8",
    "Resonance (R)": "#E45756",
    "Aromatic (A)": "#54A24B",
    "Steric (S)": "#B279A2",
    "Inductive (I_ali)": "#4C78A8",
    "Field (F_field)": "#F58518",
    "Proximity (P_prox)": "#54A24B",
    "Steric (S_ali)": "#B279A2",
}


def _component_short_label(title: str) -> str:
    if "(" in title and ")" in title:
        return title.split("(")[-1].split(")")[0]
    return title


def _safe_target_filename(name: str) -> str:

    return _safe_feature_token(str(name)).lower()


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) <= 0 or np.nanstd(yy) <= 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _rank_values(v: np.ndarray) -> np.ndarray:

    return pd.Series(np.asarray(v, dtype=float)).rank(method="average").to_numpy(dtype=float)


def prepare_component_pca_frame(
    site_df: pd.DataFrame,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    target_cols: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:


    if any(c not in site_df.columns for c in feature_cols):
        return None

    target_cols = target_cols or []
    meta_cols = [
        "site_id", "halogen", "acid_type", "base_type", "target_center_type",
        "pka_mode_inferred", "target_hii_mode", "halogenated_smiles",
        "masked_parent_smiles", "attached_is_aromatic",
    ]
    keep_cols = []
    for c in feature_cols + meta_cols + target_cols:
        if c in site_df.columns and c not in keep_cols:
            keep_cols.append(c)

    work = site_df.loc[subset_mask, keep_cols].copy()
    if len(work) < 8:
        return None

    for c in feature_cols:
        work[c] = _as_float_series(work[c])
    for c in target_cols:
        if c in work.columns:
            work[c] = _as_float_series(work[c])
    work = work.dropna(subset=feature_cols).copy()
    if len(work) < 8:
        return None

    X = work[feature_cols].to_numpy(dtype=float)
    if X.shape[0] < 8 or X.shape[1] < 2:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if np.allclose(np.nanstd(Xs, axis=0), 0):
        return None

    n_components = min(Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(Xs)
    pc_cols = [f"PC{i+1}" for i in range(coords.shape[1])]
    for i, pc in enumerate(pc_cols):
        work[pc] = coords[:, i]

                                                                                
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame({
        "feature_col": feature_cols,
        "feature_title": feature_titles,
        "feature_short": [_component_short_label(t) for t in feature_titles],
    })
    for i in range(loadings.shape[1]):
        loading_df[f"PC{i+1}_loading"] = loadings[:, i]
        loading_df[f"PC{i+1}_component_weight"] = pca.components_[i, :]
    loading_df["feature_mean_raw"] = scaler.mean_
    loading_df["feature_scale_raw"] = scaler.scale_

    explained_df = pd.DataFrame({
        "PC": pc_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": 100.0 * pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    })

    dominant_idx = np.argmax(np.abs(Xs), axis=1)
    work["dominant_component"] = np.array([feature_titles[i] for i in dominant_idx], dtype=object)
    work["dominant_component_short"] = np.array([_component_short_label(feature_titles[i]) for i in dominant_idx], dtype=object)

    return {
        "work": work,
        "X": X,
        "Xs": Xs,
        "pca": pca,
        "coords": coords,
        "pc_cols": pc_cols,
        "loading_df": loading_df,
        "explained_df": explained_df,
        "feature_cols": feature_cols,
        "feature_titles": feature_titles,
    }


def export_component_pca_tables(
    pca_data: Dict[str, Any],
    output: str,
    filename_prefix: str,
    target_cols: List[str],
) -> None:

    out_dir = fig_dir(output)
    loading_df = pca_data["loading_df"].copy()
    explained_df = pca_data["explained_df"].copy()
    work = pca_data["work"].copy()
    pc_cols = pca_data["pc_cols"]

    loading_df.to_csv(os.path.join(out_dir, f"{filename_prefix}_pca_loadings.csv"), index=False)
    explained_df.to_csv(os.path.join(out_dir, f"{filename_prefix}_pca_explained_variance.csv"), index=False)

    score_cols = []
    for c in [
        "site_id", "halogen", "acid_type", "base_type", "target_center_type",
        "pka_mode_inferred", "target_hii_mode", "dominant_component",
        "dominant_component_short", "halogenated_smiles", "masked_parent_smiles",
    ] + pc_cols + target_cols:
        if c in work.columns and c not in score_cols:
            score_cols.append(c)
    work[score_cols].to_csv(os.path.join(out_dir, f"{filename_prefix}_pca_scores.csv"), index=False)

    rows: List[Dict[str, Any]] = []
    for target in target_cols:
        if target not in work.columns:
            continue
        y = _as_float_series(work[target]).to_numpy(dtype=float)
        for pc in pc_cols:
            x = work[pc].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            pearson_r = _finite_corr(x, y)
            spearman_r = _finite_corr(_rank_values(x[mask]) if n else np.asarray([]), _rank_values(y[mask]) if n else np.asarray([]))
            slope = np.nan
            intercept = np.nan
            r2 = np.nan
            if n >= 3 and np.nanstd(x[mask]) > 0 and np.nanstd(y[mask]) > 0:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                r2 = float(pearson_r ** 2) if np.isfinite(pearson_r) else np.nan
            rows.append({
                "target": target,
                "PC": pc,
                "n": n,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "linear_r2": r2,
                "slope": float(slope) if np.isfinite(slope) else np.nan,
                "intercept": float(intercept) if np.isfinite(intercept) else np.nan,
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{filename_prefix}_pc_target_correlations.csv"), index=False)


def plot_component_pca_scree(
    pca_data: Dict[str, Any],
    subset_name: str,
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:

    explained_df = pca_data["explained_df"]
    if explained_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    x = np.arange(len(explained_df))
    y = explained_df["explained_variance_percent"].astype(float).values
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(explained_df["PC"].astype(str).tolist())
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"{subset_name}: PCA explained variance")
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}_pca_scree.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv=f"figures/{filename_prefix}_pca_explained_variance.csv",
        source_columns=["PC", "explained_variance_ratio", "explained_variance_percent", "explained_variance"],
        plot_type="pca_scree",
        subset=subset_name,
        note="Explained variance table exported by export_component_pca_tables().",
    )
    plt.close(fig)


def _plot_halogen_marker_legend(ax) -> None:

    marker_map = {"F": "o", "Cl": "s", "Br": "^", "I": "D"}
    handles = []
    for h in HALOGEN_ORDER:
        handles.append(Line2D(
            [0], [0], marker=marker_map.get(h, "o"), linestyle="None",
            markerfacecolor="#777777", markeredgecolor="#333333",
            markersize=7, label=h,
        ))
    ax.legend(handles=handles, title="Halogen", loc="best", frameon=True, fontsize=8, title_fontsize=9)


def plot_component_pca_biplot_colored_by_target(
    pca_data: Dict[str, Any],
    subset_name: str,
    target: str,
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:


    work = pca_data["work"].copy()
    loading_df = pca_data["loading_df"]
    explained = pca_data["pca"].explained_variance_ratio_
    if target not in work.columns or "PC1" not in work.columns or "PC2" not in work.columns:
        return
    work[target] = _as_float_series(work[target])
    work = work.dropna(subset=[target, "PC1", "PC2"]).copy()
    if len(work) < 5:
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    marker_map = {"F": "o", "Cl": "s", "Br": "^", "I": "D"}
    scatter_ref = None
    if "halogen" in work.columns:
        for h in HALOGEN_ORDER:
            hh = work[work["halogen"].astype(str) == h]
            if len(hh) == 0:
                continue
            sc = ax.scatter(
                hh["PC1"], hh["PC2"],
                c=hh[target].astype(float).values,
                cmap="coolwarm",
                s=28,
                alpha=0.72,
                marker=marker_map.get(h, "o"),
                edgecolors="none",
            )
            scatter_ref = sc
    else:
        scatter_ref = ax.scatter(
            work["PC1"], work["PC2"],
            c=work[target].astype(float).values,
            cmap="coolwarm",
            s=28,
            alpha=0.72,
            edgecolors="none",
        )
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax, shrink=0.82)
        cbar.set_label(target)

                                                                                
    xvals = work["PC1"].astype(float).values
    yvals = work["PC2"].astype(float).values
    xspan = float(np.nanmax(xvals) - np.nanmin(xvals)) if len(xvals) else 1.0
    yspan = float(np.nanmax(yvals) - np.nanmin(yvals)) if len(yvals) else 1.0
    span = max(xspan, yspan, 1e-6)
    load_xy = loading_df[["PC1_loading", "PC2_loading"]].to_numpy(dtype=float)
    max_norm = float(np.nanmax(np.sqrt(np.sum(load_xy ** 2, axis=1)))) if len(load_xy) else 1.0
    arrow_scale = 0.32 * span / max(max_norm, 1e-6)
    for _, row in loading_df.iterrows():
        lx = float(row.get("PC1_loading", 0.0)) * arrow_scale
        ly = float(row.get("PC2_loading", 0.0)) * arrow_scale
        label = str(row.get("feature_short", row.get("feature_title", "")))
        ax.arrow(0.0, 0.0, lx, ly, color="#222222", width=0.006 * span,
                 head_width=0.045 * span, length_includes_head=True, alpha=0.85, zorder=4)
        ax.text(lx * 1.08, ly * 1.08, label, color="#222222", fontsize=10, fontweight="bold", zorder=5)

    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
    ax.set_title(f"{subset_name}: PCA biplot colored by {target}\nn={len(work)}")
    ax.grid(alpha=0.22)
    if "halogen" in work.columns:
        _plot_halogen_marker_legend(ax)
    fig.tight_layout()
    safe_target = _safe_target_filename(target)
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}_pca_biplot_{safe_target}.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv=[
            f"figures/{filename_prefix}_pca_scores.csv",
            f"figures/{filename_prefix}_pca_loadings.csv",
            f"figures/{filename_prefix}_pca_explained_variance.csv",
        ],
        source_columns=["PC1", "PC2", target, "halogen", "PC1_loading", "PC2_loading"],
        plot_type="pca_biplot_colored_by_target",
        target=target,
        subset=subset_name,
        note="Scores provide points/colors; loadings provide arrows; explained-variance CSV provides axis percentages.",
    )
    plt.close(fig)


def plot_component_pca_bubble_by_target(
    pca_data: Dict[str, Any],
    subset_name: str,
    target: str,
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
    round_digits: int = 3,
) -> None:

    work = pca_data["work"].copy()
    explained = pca_data["pca"].explained_variance_ratio_
    if target not in work.columns or "PC1" not in work.columns or "PC2" not in work.columns:
        return
    work[target] = _as_float_series(work[target])
    work = work.dropna(subset=[target, "PC1", "PC2"]).copy()
    if len(work) < 5:
        return

    work["PC1_round"] = work["PC1"].round(round_digits)
    work["PC2_round"] = work["PC2"].round(round_digits)
    agg = work.groupby(["PC1_round", "PC2_round"], dropna=False).agg(
        n=(target, "size"),
        target_mean=(target, "mean"),
        target_median=(target, "median"),
    ).reset_index()
    if agg.empty:
        return
    max_n = max(float(agg["n"].max()), 1.0)
    sizes = 26.0 + 260.0 * np.sqrt(agg["n"].astype(float).values / max_n)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    sc = ax.scatter(
        agg["PC1_round"], agg["PC2_round"],
        s=sizes,
        c=agg["target_mean"].astype(float).values,
        cmap="coolwarm",
        alpha=0.76,
        edgecolors="#333333",
        linewidths=0.35,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.84)
    cbar.set_label(f"Mean {target}")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle=":")
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
    ax.set_title(
        f"{subset_name}: PCA bubble plot\n"
        f"size = duplicate count, color = mean {target}, n={len(work)}, unique={len(agg)}"
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    safe_target = _safe_target_filename(target)
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}_pca_bubble_{safe_target}.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv=f"figures/{filename_prefix}_pca_scores.csv",
        source_columns=["PC1", "PC2", target],
        plot_type="pca_bubble_by_target",
        target=target,
        subset=subset_name,
        note="The plot rounds PC1/PC2 coordinates before aggregation; bubble size is duplicate count and color is mean target value.",
    )
    plt.close(fig)


def plot_component_pc_vs_target(
    pca_data: Dict[str, Any],
    subset_name: str,
    target: str,
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
    max_pcs: int = 3,
) -> None:

    work = pca_data["work"].copy()
    pc_cols = [pc for pc in pca_data["pc_cols"][:max_pcs] if pc in work.columns]
    if target not in work.columns or not pc_cols:
        return
    work[target] = _as_float_series(work[target])
    work = work.dropna(subset=[target] + pc_cols).copy()
    if len(work) < 5:
        return

    fig, axes = plt.subplots(1, len(pc_cols), figsize=(5.1 * len(pc_cols), 4.5), squeeze=False)
    axes = axes.ravel()
    y = work[target].astype(float).values
    for ax, pc in zip(axes, pc_cols):
        x = work[pc].astype(float).values
        ax.scatter(x, y, s=22, alpha=0.55, edgecolors="none")
        r = _finite_corr(x, y)
        r2 = float(r ** 2) if np.isfinite(r) else np.nan
        if len(x) >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xx, slope * xx + intercept, linestyle="--", linewidth=1.6, color="#333333")
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_xlabel(pc)
        ax.set_ylabel(target)
        ax.set_title(f"{pc} vs {target}\nr={r:.3f}, R²={r2:.3f}, n={len(work)}")
        ax.grid(alpha=0.22)
    fig.suptitle(f"{subset_name}: PC-target relationships", fontsize=14, fontweight="bold")
    fig.tight_layout()
    safe_target = _safe_target_filename(target)
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}_pc_vs_{safe_target}.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv=[
            f"figures/{filename_prefix}_pca_scores.csv",
            f"figures/{filename_prefix}_pc_target_correlations.csv",
        ],
        source_columns=["PC1", "PC2", "PC3", target, "pearson_r", "linear_r2"],
        plot_type="pc_vs_target",
        target=target,
        subset=subset_name,
        note="Scores provide scatter points; correlation CSV stores the numeric PC-target correlation summary.",
    )
    plt.close(fig)


def plot_component_pca_diagnostics(
    site_df: pd.DataFrame,
    subset_name: str,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    target_cols: List[str],
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:


    pca_data = prepare_component_pca_frame(
        site_df=site_df,
        subset_mask=subset_mask,
        feature_cols=feature_cols,
        feature_titles=feature_titles,
        target_cols=target_cols,
    )
    if pca_data is None:
        return

    export_component_pca_tables(pca_data, output, filename_prefix, target_cols)
    plot_component_pca_scree(pca_data, subset_name, output, args, filename_prefix)

    for target in target_cols:
        if target not in pca_data["work"].columns:
            continue
        if _as_float_series(pca_data["work"][target]).notna().sum() < 5:
            continue
        plot_component_pca_biplot_colored_by_target(pca_data, subset_name, target, output, args, filename_prefix)
        plot_component_pca_bubble_by_target(pca_data, subset_name, target, output, args, filename_prefix)
        plot_component_pc_vs_target(pca_data, subset_name, target, output, args, filename_prefix)

def plot_component_pca_projection(
    site_df: pd.DataFrame,
    subset_name: str,
    subset_mask: pd.Series,
    feature_cols: List[str],
    feature_titles: List[str],
    output: str,
    args: argparse.Namespace,
    filename_prefix: str,
) -> None:


    if any(c not in site_df.columns for c in feature_cols):
        return

    work = site_df.loc[subset_mask].copy()
    if len(work) < 8:
        return
    for c in feature_cols:
        work[c] = _as_float_series(work[c])
    work = work.dropna(subset=feature_cols)
    if len(work) < 8:
        return

    X = work[feature_cols].to_numpy(dtype=float)
    if X.shape[0] < 8 or X.shape[1] < 3:
        return

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if np.allclose(np.nanstd(Xs, axis=0), 0):
        return

    pca = PCA(n_components=3)
    coords = pca.fit_transform(Xs)
    dominant_idx = np.argmax(np.abs(Xs), axis=1)
    dominant_labels = np.array([feature_titles[i] for i in dominant_idx], dtype=object)

    fig = plt.figure(figsize=(10.8, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    plotted = False
    for label in feature_titles:
        mask = dominant_labels == label
        if not np.any(mask):
            continue
        color = COMPONENT_PROJECTION_COLORS.get(label, "#4C78A8")
        pts = coords[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=12,
            alpha=0.60,
            color=color,
            edgecolors="none",
            label=f"{_component_short_label(label)} (n={mask.sum()})",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(f"PC 1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC 2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_zlabel(f"PC 3 ({100 * pca.explained_variance_ratio_[2]:.1f}%)")
    ax.set_title(
        f"{subset_name}: PCA projection of formula-component vectors\n"
        f"Colored by dominant component, n={len(work)}",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, 1.02), frameon=True, fontsize=9)
    fig.tight_layout()
    out_file = os.path.join(fig_dir(output), f"{filename_prefix}.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=feature_cols,
        plot_type="component_pca_projection_3d",
        subset=subset_name,
        note="PCA is recomputed from the listed formula-component columns for this 3D projection.",
    )
    plt.close(fig)


def plot_r2_summary(fit_df: pd.DataFrame, output: str, args: argparse.Namespace) -> None:


    if fit_df.empty:
        return
    metric = "full_data_r2" if "full_data_r2" in fit_df.columns else "train_r2"
    for target in fit_df["target"].dropna().unique():
        sub = fit_df[(fit_df["target"] == target) & (fit_df["subset"].isin([
            "all", "acidic_only", "acidic_plus_ambiguous", "basic_only", "ambiguous_only"
        ]))].copy()
        if sub.empty or metric not in sub.columns:
            continue
        sub[metric] = _as_float_series(sub[metric])
        sub = sub[sub[metric].notna()].copy()
        if sub.empty:
            continue
        sub["label"] = sub["subset"].astype(str) + "\n" + sub["model"].astype(str)
        sub = sub.sort_values(metric, ascending=False)
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(sub)), 5.0))
        x = np.arange(len(sub))
        ax.bar(x, sub[metric].astype(float).values)
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"].tolist(), rotation=45, ha="right")
        ax.set_ylabel("Full-data R²")
        ax.set_title(f"Final formula full-data R²: {target}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_file = os.path.join(fig_dir(output), f"acid_base_hii_full_data_r2_summary_{target}.png")
        save_figure_with_svg(fig, out_file)
        register_figure_source(
            output=output,
            figure_path=out_file,
            source_csv="acid_base_hii_fit_summary.csv",
            source_columns=["target", "subset", "model", metric, "n", "n_features"],
            plot_type="full_data_r2_summary",
            target=target,
            fit_kind="full_data",
            note="Primary R² summary using the full-data metric selected inside plot_r2_summary().",
        )
        plt.close(fig)


def plot_mode_counts(site_df: pd.DataFrame, output: str, args: argparse.Namespace) -> None:
    if "pka_mode_inferred" not in site_df.columns:
        return
    counts = site_df["pka_mode_inferred"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(counts))
    ax.bar(x, counts.values)
    ax.set_xticks(x); ax.set_xticklabels(counts.index.tolist(), rotation=30, ha="right")
    ax.set_ylabel("Halogen sites")
    ax.set_title("Inferred pKa mode distribution")
    for xi, yi in zip(x, counts.values):
        ax.text(xi, yi, str(int(yi)), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_file = os.path.join(fig_dir(output), "pka_mode_counts.png")
    save_figure_with_svg(fig, out_file)
    register_figure_source(
        output=output,
        figure_path=out_file,
        source_csv="counterfactual_mask_acid_base_hii_sites.csv",
        source_columns=["pka_mode_inferred"],
        plot_type="pka_mode_counts",
        note="Counts are computed from the inferred pKa mode column in the counterfactual site table.",
    )
    plt.close(fig)


def make_plots(site_df: pd.DataFrame, fit_df: pd.DataFrame, pred_df: pd.DataFrame, targets: List[str], output: str, args: argparse.Namespace) -> None:
    reset_figure_source_manifest()
    plot_mode_counts(site_df, output, args)
    mode = site_df.get("pka_mode_inferred", pd.Series("", index=site_df.index)).astype(str).str.lower()
    acidic_mask = mode.eq("acidic") | mode.str.contains("fallback_acid", case=False, na=False)
    dist_targets = [t for t in ["delta_pka_true", "delta_pka_pred"] if t in targets]
    pca_targets = [t for t in ["delta_pka_true", "delta_pka_pred", "delta_error"] if t in targets and t in site_df.columns]

    acid_feature_cols = [
        "acid_hii_inductive_selected",
        "acid_hii_resonance_selected",
        "acid_hii_aromatic_selected",
        "acid_hii_steric_selected",
    ]
    acid_feature_titles = ["Inductive (I)", "Resonance (R)", "Aromatic (A)", "Steric (S)"]

    for target in targets:
        plot_delta_vs_hii(site_df, target, "target_hii_total", os.path.join(fig_dir(output), f"{target}_vs_target_hii_total.png"), f"{target} vs target-center HII")
        if "acid_hii_total_mean" in site_df.columns:
            plot_delta_vs_hii(site_df, target, "acid_hii_total_mean", os.path.join(fig_dir(output), f"{target}_vs_acid_hii_total.png"), f"{target} vs unified acid-directed HII")
        if "base_hii_total_mean" in site_df.columns:
            plot_delta_vs_hii(site_df, target, "base_hii_total_mean", os.path.join(fig_dir(output), f"{target}_vs_base_hii_total.png"), f"{target} vs base-directed HII")
        plot_best_parity(pred_df, fit_df, target, output, args)

                                                                                
                                     
    for target in dist_targets:
        plot_halogen_delta_distribution_panel(site_df, target, output, args)
        plot_position_delta_distribution_panel(site_df, target, output, args)
        plot_formula_distribution_panels(
            site_df=site_df,
            target=target,
            subset_name="Unified acidic samples",
            subset_mask=acidic_mask,
            feature_cols=acid_feature_cols,
            feature_titles=acid_feature_titles,
            output=output,
            args=args,
        )
        plot_iras_horizontal_delta_distribution_panel(
            site_df=site_df,
            fit_df=fit_df,
            target=target,
            subset_name="Unified acidic samples",
            subset_mask=acidic_mask,
            feature_cols=acid_feature_cols,
            feature_titles=acid_feature_titles,
            output=output,
            args=args,
            filename_prefix="unified_acidic",
        )

                                                                           
                                                                            
                               
    plot_formula_parameter_range_distribution_panel(
        site_df=site_df,
        subset_name="Unified acidic samples",
        subset_mask=acidic_mask,
        feature_cols=acid_feature_cols,
        feature_titles=acid_feature_titles,
        output=output,
        args=args,
        filename_prefix="unified_acidic",
    )

                                                                       
    plot_component_pca_projection(
        site_df=site_df,
        subset_name="Unified acidic samples",
        subset_mask=acidic_mask,
        feature_cols=acid_feature_cols,
        feature_titles=acid_feature_titles,
        output=output,
        args=args,
        filename_prefix="unified_acidic_component_pca_projection",
    )

    plot_component_pca_diagnostics(
        site_df=site_df,
        subset_name="Unified acidic samples",
        subset_mask=acidic_mask,
        feature_cols=acid_feature_cols,
        feature_titles=acid_feature_titles,
        target_cols=pca_targets,
        output=output,
        args=args,
        filename_prefix="unified_acidic_component",
    )
    plot_r2_summary(fit_df, output, args)
    write_figure_source_manifest(output)


                                                                               
        
                                                                               

def write_report(
    site_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output: str,
    metadata: Dict[str, Any],
    acid_unified_df: Optional[pd.DataFrame] = None,
    final_formula_df: Optional[pd.DataFrame] = None,
    final_formula_by_subset_df: Optional[pd.DataFrame] = None,
    all_formula_df: Optional[pd.DataFrame] = None,
    final_parameter_df: Optional[pd.DataFrame] = None,
    final_by_subset_parameter_df: Optional[pd.DataFrame] = None,
    formula_explanation_df: Optional[pd.DataFrame] = None,
) -> None:
    lines = []
    lines.append("# Acid/base-aware counterfactual HII analysis")
    lines.append("")
    lines.append("This analysis freezes the trained GNN and estimates halogen-induced pKa shifts through counterfactual masking:")
    lines.append("")
    lines.append("\\[\\Delta pK_a = pK_a(R-X)-pK_a(R-H)\\]")
    lines.append("")
    lines.append("It computes three HII variants per halogen site:")
    lines.append("")
    lines.append("- **acid-directed HII**: HII toward acidic centers.")
    lines.append("- **base-directed HII**: HII toward basic pKaH centers.")
    lines.append("- **target-center HII**: acid or base HII selected according to the pKa label mode.")
    lines.append("")
    lines.append("The primary quantitative result is the final fitted formula and its **full-data R²**. Train/CV metrics are retained only as secondary diagnostics.")
    lines.append("")
    lines.append("## Metadata")
    for k, v in metadata.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Data summary")
    lines.append(f"- Halogen sites: {len(site_df)}")
    lines.append(f"- Unique halogenated molecules: {site_df['halogenated_smiles'].nunique() if 'halogenated_smiles' in site_df else 'NA'}")
    lines.append(f"- Unique masked parents: {site_df['masked_parent_smiles'].nunique() if 'masked_parent_smiles' in site_df else 'NA'}")
    if "delta_pka_true" in site_df.columns:
        lines.append(f"- Experimental parent matches: {site_df['delta_pka_true'].notna().sum()}")
    if "pka_mode_inferred" in site_df.columns:
        lines.append("")
        lines.append("### pKa mode counts")
        lines.append(site_df["pka_mode_inferred"].astype(str).value_counts().to_frame("n").to_markdown())
    lines.append("")
    lines.append("## Main result: final formula + full-data R²")
    lines.append("")
    if final_formula_df is not None and not final_formula_df.empty:
        show = [c for c in [
            "target", "subset", "model", "n", "n_features",
            "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule", "formula",
        ] if c in final_formula_df.columns]
        lines.append(final_formula_df[show].to_markdown(index=False))
    else:
        lines.append("No final formula row was produced. Check sample sizes and feature availability.")
    lines.append("")
    lines.append("CSV outputs: `final_formula_full_data_r2.csv`, `final_formula_by_subset_full_data_r2.csv`, `all_formulas_full_data_r2.csv`, `final_formula_parameters.csv`, `final_formula_by_subset_parameters.csv`, and `formula_parameter_explanations.csv`.")
    lines.append("The exported final formula is post-processed exactly as requested: `final_formula = fitted_formula / 0.64109 + 0.210521`. The formula coefficients and full-data fitted values use this transformed formula, but fit-summary R²/RMSE/MAE are kept from the original fitted formula before the transform, so R² does not change.")
    lines.append("Requested full-data application tables are written to `pred_full_data.csv` and `true_full_data.csv` with aliases `acidic_pred_full_data_final_formula.csv` and `acidic_true_full_data_final_formula.csv`; both remove duplicate rows with the same SMILES and same delta_pka.")
    lines.append("Explicit final-HII-formula aliases are also written to `final_hii_formula_summary.csv`, `final_hii_formula_parameters.csv`, and `final_hii_formula_with_parameters.md`; these include formula text, parameter values, and parameter meanings.")
    lines.append("When `--save-plots` is used, every generated figure is saved as a 300-dpi PNG and as a same-name SVG. `figure_source_data_map.csv` records every generated PNG/SVG figure and the source CSV file(s)/columns used to create it. The full-data best-parity plots also write `best_parity_<target>_full_data_source_rows.csv`, including `best_parity_delta_pka_pred_full_data_source_rows.csv` and `best_parity_delta_pka_true_full_data_source_rows.csv`.")
    lines.append("Important: `best_parity_delta_pka_true_full_data.png` applies the HII formula learned from `delta_pka_pred` to experimental `delta_pka_true`; it is not refit on the experimental target. The numeric summary for this transfer is written to `transferred_pred_formula_to_true_summary.csv`.")
    lines.append("")

    if final_parameter_df is not None and not final_parameter_df.empty:
        lines.append("## Main final formula: all parameter values and explanations")
        lines.append("")
        for (target, subset, model), sub in final_parameter_df.groupby(["target", "subset", "model"], dropna=False):
            lines.append(f"### target={target}, subset={subset}, model={model}")
            lines.append("")
            show = [c for c in ["term_order", "parameter_kind", "display_name", "coefficient", "activation_rule", "explanation"] if c in sub.columns]
            lines.append(sub[show].to_markdown(index=False))
            lines.append("")

    if final_formula_by_subset_df is not None and not final_formula_by_subset_df.empty:
        lines.append("## Subset final formulas with full-data R²")
        lines.append("")
        show = [c for c in [
            "target", "subset", "model", "n", "n_features",
            "full_data_r2", "full_data_rmse", "full_data_mae", "selection_rule", "formula",
        ] if c in final_formula_by_subset_df.columns]
        lines.append(final_formula_by_subset_df[show].to_markdown(index=False))
        lines.append("")
    if final_by_subset_parameter_df is not None and not final_by_subset_parameter_df.empty:
        lines.append("Detailed subset parameter values are written to `final_formula_by_subset_parameters.csv` and `final_formula_with_parameters.md`.")
        lines.append("")
    if formula_explanation_df is not None and not formula_explanation_df.empty:
        lines.append("## Formula symbol glossary")
        lines.append("")
        show = [c for c in ["raw_feature", "display_name", "activation_rule", "explanation"] if c in formula_explanation_df.columns]
        lines.append(formula_explanation_df[show].to_markdown(index=False))
        lines.append("")

    lines.append("## HII fit summary, full-data first; train/CV are diagnostics")
    cols = [c for c in [
        "target", "subset", "model", "n", "n_features",
        "full_data_r2", "full_data_rmse", "full_data_mae",
        "train_r2", "cv_r2", "train_rmse", "cv_rmse", "train_mae", "cv_mae",
    ] if c in fit_df.columns]
    if cols:
        sort_col = "full_data_r2" if "full_data_r2" in fit_df.columns else "train_r2"
        lines.append(fit_df[cols].sort_values(["target", sort_col], ascending=[True, False]).to_markdown(index=False))
    lines.append("")

    if acid_unified_df is not None and not acid_unified_df.empty:
        lines.append("## Unified acid R² summary")
        lines.append("")
        show = [c for c in [
            "target", "acid_report", "subset", "model", "recommended_use", "n",
            "full_data_r2", "full_data_rmse", "full_data_mae",
            "train_r2", "cv_r2", "cv_rmse", "cv_mae", "note"
        ] if c in acid_unified_df.columns]
        sort_col = "full_data_r2" if "full_data_r2" in acid_unified_df.columns else "cv_r2"
        sort_cols = [c for c in ["target", sort_col] if c in acid_unified_df.columns]
        if show:
            lines.append(acid_unified_df[show].sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True).to_markdown(index=False))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- If acid-only and base-only formulas have different coefficients/full-data R², acid and base pKa labels should not be mixed into one scalar interpretation.")
    lines.append("- If the mixed interaction formula has higher full-data R², pKa mode changes how HII components transmit into ΔpKa.")
    lines.append("- The retained candidate family is `acid_fixed_triple_scaffold__...`: ΔpKa = b0 + bI·I + bR·R + bA·A + bS·S + β1·D1 + β2·D2 + β3·D3. All halogens and acid-center types share the same I/R/A/S coefficients.")
    lines.append("- D1/D2/D3 are selected from the 10 raw RDKit scaffold descriptors computed from masked parent R-H. `scaffold_triple_r2_ranking.csv` ranks all C(10,3)=120 descriptor triples by full-data R² and CV diagnostics.")
    lines.append("- Acidic samples now use one unified acid HII formula: `acid_hii_inductive_selected`, `acid_hii_resonance_selected`, `acid_hii_aromatic_selected`, and `acid_hii_steric_selected`, with no aromatic/non-aromatic split for fitting or R² reporting.")
    lines.append("- `final_formula_full_data_r2.csv` is the main result table: one final formula per target plus its full-data R².")
    lines.append("- Final reported fitted values are calculated from `final_formula = fitted_formula / 0.64109 + 0.210521`; fit-summary R² is intentionally kept from the original fitted formula before this affine transform, so R² is unchanged.")
    lines.append("- `pred_full_data.csv` and `true_full_data.csv` are the requested acidic full-data tables; both are de-duplicated by same SMILES + same delta_pka before metrics/plots/source-row exports.")
    lines.append("- `acid_unified_r2_summary.csv` reports R² for the single `acidic_only` subset.")
    lines.append("- When `--save-plots` is used, the script writes `unified_acidic_parameter_range_distribution.png`, a same-format distribution plot of the four acid HII parameter ranges.")
    lines.append("- It also writes `position_distribution_<target>.png`, matching the halogen-distribution format but grouped by ortho/meta/para/same_ring_other/same_aromatic_system/aliphatic/unknown; empty categories are safely dropped before tick labels are styled.")
    lines.append("- It also writes `unified_acidic_horizontal_iras_distribution_<target>.png`, plus `unified_acidic_horizontal_iras_distribution_<target>_source_data.csv` and `_source_data_wide.csv`; the long CSV has one row per sample × formula component. All plotted x-values are coefficient-weighted contributions from the explicit fitted formula: I=-4.01896*I_acid, R=0.925079*R_acid, A=-0.889082*A_acid, S=-1.62732*S_acid, and Scaffold=0.0023094*TPSA + 0.138223*MolLogP + 0.126668*FractionCSP3.")
    lines.append("- Parity plots are now exported for `full_data`, `train`, and `cv`; the `full_data` version is the primary fitted-on-all-data figure.")
    lines.append("- Every generated figure is saved twice: a 300-dpi PNG and a same-name SVG vector file in the `figures/` directory.")
    lines.append("- `figure_source_data_map.csv` explains which output CSV file(s) and columns were used for each generated figure.")
    lines.append("- `best_parity_delta_pka_pred_full_data_source_rows.csv` contains the exact rows used by `best_parity_delta_pka_pred_full_data.png`, including the raw input-file halogenated SMILES/pKa and the pre-halogenation parent SMILES/pKa when that parent exists in the original CSV.")
    lines.append("- `best_parity_delta_pka_true_full_data_source_rows.csv` contains the exact rows used by `best_parity_delta_pka_true_full_data.png`; the y-axis values come from the `delta_pka_pred` HII formula, while the x-axis observed values are experimental `delta_pka_true`.")
    lines.append("- `final_hii_formula_parameters.csv` gives the final HII formula parameters, coefficient values, activation rules, and meanings in one table.")
    lines.append("- PCA diagnostics include 2D PC1/PC2 biplots with loading arrows, target-colored PCA maps, duplicate-aware PCA bubble plots, scree plots, PCA scores/loadings CSV files, and PC-target correlation tables for the unified acidic component set.")
    lines.append("- `target_hii_*` columns are the recommended columns for downstream halogen-specific analysis because they use acid/base-directed HII according to the pKa label mode.")
    with open(os.path.join(output, "acid_base_hii_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


                                                                               
      
                                                                               

def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    cfg, task_names, model, ckpt, cfg_source, arch_changes, load_msg = _load_cfg_and_model(args)
    input_df, target_task, true_col = load_input_table(cfg, args, task_names)
    input_csv_path = input_df.attrs.get(
        "input_csv_path",
        args.input_csv or os.path.join(cfg.DATA.DATA_PATH, "raw", f"{cfg.DATA.DATASET}.csv"),
    )

    model_target_task = target_task if target_task in task_names else task_names[0]
    if target_task not in task_names:
        warnings.warn(f"target_task {target_task!r} not in cfg task names {task_names}; using model output task {model_target_task!r}.")

    true_map = build_true_value_map(input_df, true_col)
    site_df = build_halogen_mask_sites(input_df, args, true_col)
    if site_df.empty:
        raise RuntimeError("No halogen mask sites found.")
    site_df = annotate_acid_partitions(site_df)
    site_df = add_scaffold_background_scores(site_df)
    site_df = annotate_site_rows_with_original_input_data(
        site_df=site_df,
        input_df=input_df,
        args=args,
        true_col=true_col,
        input_csv_path=input_csv_path,
    )

    all_smiles = list(site_df["halogenated_smiles"].unique()) + list(site_df["masked_parent_smiles"].unique())
    batch_size = int(args.batch_size or cfg.DATA.BATCH_SIZE)
    pred_map = predict_smiles_list(model, cfg, all_smiles, task_names, model_target_task, batch_size=batch_size)
    site_df["pka_pred_halogenated"] = site_df["halogenated_smiles"].map(pred_map)
    site_df["pka_pred_parent_masked"] = site_df["masked_parent_smiles"].map(pred_map)
    site_df["delta_pka_pred"] = site_df["pka_pred_halogenated"] - site_df["pka_pred_parent_masked"]

    if true_map:
        site_df["pka_true_parent_matched"] = site_df["masked_parent_smiles"].map(true_map)
        if "pka_true_halogenated" in site_df.columns:
            site_df["delta_pka_true"] = site_df["pka_true_halogenated"] - site_df["pka_true_parent_matched"]
            site_df["delta_error"] = site_df["delta_pka_pred"] - site_df["delta_pka_true"]

    fit_targets = ["delta_pka_pred"]
    plot_targets = ["delta_pka_pred"]
    if "delta_pka_true" in site_df.columns and site_df["delta_pka_true"].notna().sum() >= args.min_group_size:
        plot_targets.append("delta_pka_true")
    if "delta_error" in site_df.columns and site_df["delta_error"].notna().sum() >= args.min_group_size:
        fit_targets.append("delta_error")
        plot_targets.append("delta_error")

    fit_df, pred_df = fit_all_models(site_df, fit_targets, args)
    pred_df = enrich_fit_predictions_with_site_metadata(pred_df, site_df)

                                                                      
                                                                            
                                                                 
    final_full_data_application_df = export_acidic_final_formula_full_data_tables(
        pred_df=pred_df,
        fit_df=fit_df,
        output=args.output,
    )

    all_formula_df, final_formula_df, final_formula_by_subset_df = export_final_formula_outputs(fit_df, args.output, args)
    final_parameter_df, final_by_subset_parameter_df, formula_explanation_df = export_final_formula_parameter_details(
        fit_df=fit_df,
        final_formula_df=final_formula_df,
        final_formula_by_subset_df=final_formula_by_subset_df,
        output_dir=args.output,
        args=args,
    )
    final_hii_formula_df, final_hii_parameter_df = export_final_hii_formula_outputs(
        final_formula_df=final_formula_df,
        final_formula_by_subset_df=final_formula_by_subset_df,
        final_parameter_df=final_parameter_df,
        final_by_subset_parameter_df=final_by_subset_parameter_df,
        output_dir=args.output,
    )

    input_df.to_csv(os.path.join(args.output, "standardized_input_molecules.csv"), index=False)
    site_df.to_csv(os.path.join(args.output, "counterfactual_mask_acid_base_hii_sites.csv"), index=False)
    pd.DataFrame({"smiles": list(pred_map.keys()), f"{model_target_task}_pred": list(pred_map.values())}).to_csv(
        os.path.join(args.output, "counterfactual_model_predictions.csv"), index=False
    )
    fit_df.to_csv(os.path.join(args.output, "acid_base_hii_fit_summary.csv"), index=False)
    pred_df.to_csv(os.path.join(args.output, "acid_base_hii_fit_predictions.csv"), index=False)
                                                                   
    all_formula_df.to_csv(os.path.join(args.output, "all_formulas_full_data_r2.csv"), index=False)
    final_formula_df.to_csv(os.path.join(args.output, "final_formula_full_data_r2.csv"), index=False)
    final_formula_by_subset_df.to_csv(os.path.join(args.output, "final_formula_by_subset_full_data_r2.csv"), index=False)
    scaffold_triple_ranking_df = export_scaffold_triple_r2_ranking(fit_df, args.output)
    acid_fixed_coef_df = export_acid_fixed_coefficients(fit_df, args.output)
    acid_scalar_diag_df = export_acid_single_scalar_diagnostics(fit_df, args.output)
    acid_unified_df = export_acid_unified_r2_summary(fit_df, pred_df, args.output)
    acid_partition_df = export_acid_partition_summary(site_df, args.output, plot_targets)
    save_group_summaries(site_df, args.output, args)

    transfer_true_summary_df = pd.DataFrame()
    if "delta_pka_true" in site_df.columns and site_df["delta_pka_true"].notna().sum() >= args.min_group_size:
        transfer_true_summary_df = export_transferred_pred_formula_to_true_outputs(
            pred_df=pred_df,
            fit_df=fit_df,
            output=args.output,
            args=args,
            save_plot=False,
        )

    if args.save_plots:
        make_plots(site_df, fit_df, pred_df, plot_targets, args.output, args)
                                                                                
                                                                  
        if "delta_pka_true" in site_df.columns and site_df["delta_pka_true"].notna().sum() >= args.min_group_size:
            _ = export_transferred_pred_formula_to_true_outputs(
                pred_df=pred_df,
                fit_df=fit_df,
                output=args.output,
                args=args,
                save_plot=True,
            )
            write_figure_source_manifest(args.output)

    metadata = {
        "checkpoint": args.checkpoint,
        "config_source": cfg_source,
        "architecture_adjustments": arch_changes,
        "load_message": str(load_msg),
        "input_rows": len(input_df),
        "input_csv_path": input_csv_path,
        "counterfactual_sites": len(site_df),
        "target_task_csv": target_task,
        "target_task_model": model_target_task,
        "true_col": true_col,
        "pka_type_col": args.pka_type_col,
        "hii_aggregation": args.hii_aggregation,
        "hii_weight_tau": args.hii_weight_tau,
        "ambiguous_policy": args.ambiguous_policy,
        "unknown_policy": getattr(args, "unknown_policy", "zero"),
        "fit_model": args.fit_model,
        "cv": args.cv,
        "group_col": args.group_col,
        "main_formula_model": getattr(args, "main_formula_model", "auto_best_full_data"),
        "formula_precision": getattr(args, "formula_precision", 6),
        "final_formula_transform": FINAL_FORMULA_TRANSFORM,
        "final_formula_scale_divisor": FINAL_FORMULA_SCALE,
        "final_formula_offset": FINAL_FORMULA_OFFSET,
        "fit_metrics_r2_kept_from_original_fitted_formula": True,
        "postfit_transform_refit": False,
        "pred_full_data_csv": PRED_FULL_DATA_CSV,
        "true_full_data_csv": TRUE_FULL_DATA_CSV,
        "acidic_pred_full_data_final_formula_csv": ACIDIC_PRED_FULL_DATA_CSV,
        "acidic_true_full_data_final_formula_csv": ACIDIC_TRUE_FULL_DATA_CSV,
        "full_data_dedup_rule": "same SMILES + same delta_pka",
        "final_formula_full_data_application_summary": FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV,
        "final_full_data_application_rows": int(len(final_full_data_application_df)) if "final_full_data_application_df" in locals() else 0,
        "figure_source_manifest": FIGURE_SOURCE_MANIFEST if args.save_plots else None,
        "figure_png_dpi": FIGURE_EXPORT_DPI if args.save_plots else None,
        "figure_svg_export": bool(args.save_plots),
        "figure_export_formats": "png_300dpi|svg" if args.save_plots else None,
        "best_parity_delta_pka_pred_full_data_source_rows": "best_parity_delta_pka_pred_full_data_source_rows.csv" if args.save_plots else None,
        "best_parity_delta_pka_true_full_data_source_rows": "best_parity_delta_pka_true_full_data_source_rows.csv" if args.save_plots else None,
        "best_parity_delta_pka_true_full_data_formula_source": "delta_pka_pred_full_data_formula_applied_without_refit" if ("delta_pka_true" in site_df.columns and site_df["delta_pka_true"].notna().sum() >= args.min_group_size) else None,
        "transferred_pred_formula_to_true_summary": "transferred_pred_formula_to_true_summary.csv" if ("delta_pka_true" in site_df.columns and site_df["delta_pka_true"].notna().sum() >= args.min_group_size) else None,
        "final_hii_formula_summary": "final_hii_formula_summary.csv",
        "final_hii_formula_parameters": "final_hii_formula_parameters.csv",
        "final_hii_formula_report": "final_hii_formula_with_parameters.md",
        "final_formula_rows": int(len(final_formula_df)) if "final_formula_df" in locals() else 0,
        "final_formula_by_subset_rows": int(len(final_formula_by_subset_df)) if "final_formula_by_subset_df" in locals() else 0,
        "all_formula_rows": int(len(all_formula_df)) if "all_formula_df" in locals() else 0,
        "final_formula_parameter_rows": int(len(final_parameter_df)) if "final_parameter_df" in locals() else 0,
        "final_formula_by_subset_parameter_rows": int(len(final_by_subset_parameter_df)) if "final_by_subset_parameter_df" in locals() else 0,
        "formula_explanation_rows": int(len(formula_explanation_df)) if "formula_explanation_df" in locals() else 0,
        "final_hii_formula_rows": int(len(final_hii_formula_df)) if "final_hii_formula_df" in locals() else 0,
        "final_hii_formula_parameter_rows": int(len(final_hii_parameter_df)) if "final_hii_parameter_df" in locals() else 0,
        "model_task_names": task_names,
        "model_mode_three_level": bool(hasattr(cfg.MODEL, "THREE_LEVEL") and cfg.MODEL.THREE_LEVEL.ENABLE),
        "model_brics": bool(getattr(cfg.MODEL, "BRICS", False)),
        "acid_fixed_coefficients_rows": int(len(acid_fixed_coef_df)) if "acid_fixed_coef_df" in locals() else 0,
        "acid_single_scalar_diagnostics_rows": int(len(acid_scalar_diag_df)) if "acid_scalar_diag_df" in locals() else 0,
        "acid_unified_r2_summary_rows": int(len(acid_unified_df)) if "acid_unified_df" in locals() else 0,
        "acid_partition_summary_rows": int(len(acid_partition_df)) if "acid_partition_df" in locals() else 0,
        "acid_hii_formula": "acid_fixed_IRAS_plus_scaffold_descriptor_triples",
        "retained_formula": "delta_pka = b0 + bI*I + bR*R + bA*A + bS*S + beta1*D1 + beta2*D2 + beta3*D3",
        "scaffold_terms": "selected_from_scaffold_descriptor_columns_by_best_full_data_r2",
        "scaffold_descriptor_columns": SCAFFOLD_DESCRIPTOR_COLS,
        "scaffold_descriptor_combination_size": SCAFFOLD_TRIPLE_SIZE,
        "scaffold_descriptor_candidate_count": int(math.comb(len(SCAFFOLD_TRIPLE_DESCRIPTOR_COLS), SCAFFOLD_TRIPLE_SIZE)),
        "scaffold_triple_r2_ranking": "scaffold_triple_r2_ranking.csv",
        "scaffold_triple_ranking_rows": int(len(scaffold_triple_ranking_df)) if "scaffold_triple_ranking_df" in locals() else 0,
        "non_aromatic_acid_hii": "disabled_legacy_columns_zeroed",
    }
    with open(os.path.join(args.output, "acid_base_hii_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    write_report(
        site_df,
        fit_df,
        args.output,
        metadata,
        acid_unified_df=acid_unified_df,
        final_formula_df=final_formula_df,
        final_formula_by_subset_df=final_formula_by_subset_df,
        all_formula_df=all_formula_df,
        final_parameter_df=final_parameter_df,
        final_by_subset_parameter_df=final_by_subset_parameter_df,
        formula_explanation_df=formula_explanation_df,
    )

    print("=" * 80)
    print("Acid/base-aware counterfactual HII analysis complete")
    print(f"Output: {args.output}")
    print(f"Counterfactual sites: {len(site_df)}")
    print(f"Unique halogenated molecules: {site_df['halogenated_smiles'].nunique()}")
    print(f"Unique masked parents: {site_df['masked_parent_smiles'].nunique()}")
    if all(c in site_df.columns for c in SCAFFOLD_TRIPLE_DESCRIPTOR_COLS):
        print("Raw scaffold descriptors from masked parent R-H used for triple search:")
        for c in SCAFFOLD_TRIPLE_DESCRIPTOR_COLS:
            vals = pd.to_numeric(site_df[c], errors="coerce").dropna()
            if len(vals):
                print(f"  {feature_label(c)}: mean={vals.mean():.4f}, std={vals.std(ddof=0):.4f}, n={len(vals)}")
    if "delta_pka_true" in site_df.columns:
        print(f"Experimental parent matches: {site_df['delta_pka_true'].notna().sum()}")
    if "pka_mode_inferred" in site_df.columns:
        print("pKa mode counts:")
        print(site_df["pka_mode_inferred"].astype(str).value_counts().to_string())
    if "acid_partition" in site_df.columns:
        print("Acid partition counts:")
        print(site_df["acid_partition"].astype(str).value_counts().to_string())

    print("Main result: required scaffold-corrected acid-context formula + full-data R²")
    if 'final_formula_df' in locals() and isinstance(final_formula_df, pd.DataFrame) and len(final_formula_df):
        show_cols = [c for c in ["target", "subset", "model", "n", "full_data_r2", "formula"] if c in final_formula_df.columns]
        print(final_formula_df[show_cols].to_string(index=False, max_colwidth=220))
    else:
        print("  No final formula rows generated. Check sample sizes and feature columns.")
    print(f"Formula CSV: {os.path.join(args.output, 'final_formula_full_data_r2.csv')}")
    print(f"Parameter CSV: {os.path.join(args.output, 'final_formula_parameters.csv')}")
    print(f"Parameter report: {os.path.join(args.output, 'final_formula_with_parameters.md')}")
    print(f"Final HII formula CSV: {os.path.join(args.output, 'final_hii_formula_summary.csv')}")
    print(f"Final HII formula parameter CSV: {os.path.join(args.output, 'final_hii_formula_parameters.csv')}")
    print(f"Final HII formula report: {os.path.join(args.output, 'final_hii_formula_with_parameters.md')}")
    print(f"Final formula transform: {FINAL_FORMULA_TRANSFORM}")
    print("R² policy: unchanged from original fitted formula before /0.64109 + 0.210521; no refit and no post-transform R² recomputation in fit summaries.")
    print(f"Pred full_data CSV: {os.path.join(args.output, PRED_FULL_DATA_CSV)}")
    print(f"True full_data CSV: {os.path.join(args.output, TRUE_FULL_DATA_CSV)}")
    print(f"Full_data application summary CSV: {os.path.join(args.output, FINAL_FULL_DATA_APPLICATION_SUMMARY_CSV)}")
    print("Full_data de-duplication rule: same SMILES + same delta_pka")
    if 'final_full_data_application_df' in locals() and isinstance(final_full_data_application_df, pd.DataFrame) and len(final_full_data_application_df):
        show_cols = [c for c in ["table", "n", "r2", "rmse", "mae", "source_model", "source_subset", "dedup_rule"] if c in final_full_data_application_df.columns]
        print("Final formula full_data application:")
        print(final_full_data_application_df[show_cols].to_string(index=False, max_colwidth=160))
    print(f"All formula CSV: {os.path.join(args.output, 'all_formulas_full_data_r2.csv')}")
    print(f"Scaffold triple R² ranking CSV: {os.path.join(args.output, 'scaffold_triple_r2_ranking.csv')}")
    if args.save_plots:
        print(f"Figure source map CSV: {os.path.join(args.output, FIGURE_SOURCE_MANIFEST)}")
        print("Figure export: every generated figure is saved as 300-dpi PNG plus same-name SVG")
        print(f"Best parity delta_pka_pred full-data source rows CSV: {os.path.join(args.output, 'best_parity_delta_pka_pred_full_data_source_rows.csv')}")
        print(f"Best parity delta_pka_true full-data source rows CSV: {os.path.join(args.output, 'best_parity_delta_pka_true_full_data_source_rows.csv')}")
        print("Best parity delta_pka_true full-data uses the delta_pka_pred HII formula without experimental-target refit.")
        print(f"Best parity delta_pka_pred full-data formula parameters CSV: {os.path.join(args.output, 'best_parity_delta_pka_pred_full_data_formula_parameters.csv')}")
        print(f"Best parity delta_pka_true full-data formula parameters CSV: {os.path.join(args.output, 'best_parity_delta_pka_true_full_data_formula_parameters.csv')}")
    if 'transfer_true_summary_df' in locals() and isinstance(transfer_true_summary_df, pd.DataFrame) and len(transfer_true_summary_df):
        row = transfer_true_summary_df.iloc[0]
        print("Transferred pred-formula to true:")
        print(f"  formula_source = {row.get('formula_source', 'delta_pka_pred')}")
        print(f"  evaluation_target = {row.get('evaluation_target', 'delta_pka_true')}")
        print(f"  R² = {float(row['r2']):.4f}" if pd.notna(row.get('r2', np.nan)) else "  R² = nan")
        print(f"  RMSE = {float(row['rmse']):.4f}" if pd.notna(row.get('rmse', np.nan)) else "  RMSE = nan")
        print(f"  MAE = {float(row['mae']):.4f}" if pd.notna(row.get('mae', np.nan)) else "  MAE = nan")
        print(f"  n = {int(row['n'])}" if pd.notna(row.get('n', np.nan)) else "  n = 0")
        print(f"  source_model = {row.get('source_model', '')} ({row.get('source_subset', '')})")
        print(f"  source_selection_metric = {row.get('source_selection_metric', '')}")
        print(f"  note = {row.get('note', '')}")
        print(f"Transferred summary CSV: {os.path.join(args.output, 'transferred_pred_formula_to_true_summary.csv')}")

    if 'final_parameter_df' in locals() and isinstance(final_parameter_df, pd.DataFrame) and len(final_parameter_df):
        print("Final formula parameters:")
        show_cols = [c for c in ["target", "term_order", "display_name", "coefficient", "activation_rule", "explanation"] if c in final_parameter_df.columns]
        print(final_parameter_df[show_cols].to_string(index=False, max_colwidth=140))

    print("Formula CV diagnostic for the retained model")
    for t in fit_targets:
        sub = fit_df[(fit_df["target"] == t) & fit_df["cv_r2"].notna()]
        if len(sub):
            row = sub.iloc[0]
            print(f"  {t}: {row['model']} ({row['subset']}), CV R²={row['cv_r2']:.4f}, n={int(row['n'])}")

    print("=" * 80)


if __name__ == "__main__":
    main()
