# ============================================================
# train_final_model.py
#
# FINAL FROZEN WEIGHTED CATBOOST MODEL
#
# Purpose:
#   Train the final selected CatBoost model on the complete
#   cleaned SEERA dataset for subsequent XAI analysis.
#
# Frozen configuration:
#   - 50 predictive features
#   - log1p target transformation
#   - Optuna-selected hyperparameters
#   - 3x sample weight for Actual Effort < 5,000 hours
#
# NOTE:
#   This model is NOT used to report the LOGO-CV performance.
#   The LOGO-CV evaluation was already performed separately.
#
#   This full-data model is saved specifically so that SHAP
#   and DiCE can explain the final selected model.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# train_final_model.py is expected to be inside:
#
# Project practical/
#     model/
#         train_final_model.py
#
# Therefore parents[1] = Project practical/

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    /"processed"
    / "SEERA_cleaned.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "saved_model"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "final_weighted_catboost.cbm"
)


METADATA_PATH = (
    MODEL_DIR
    / "model_metadata.json"
)


# ============================================================
# 3. FROZEN SETTINGS
# ============================================================

RANDOM_SEED = 42

WEIGHT_THRESHOLD = 5000

LOW_EFFORT_WEIGHT = 3.0


# Frozen parameters selected through Optuna
FINAL_PARAMS = {

    "depth":
        3,

    "learning_rate":
        0.04124173274946173,

    "iterations":
        600,

    "l2_leaf_reg":
        13.500717070418117,

    "random_strength":
        0.41882193959396974,

    "bagging_temperature":
        1.9187145799605694
}


target_col = (
    "Effort_Actual effort"
)

group_col = (
    "General Information_Organization id"
)

id_col = (
    "General Information_ProjID"
)


# ============================================================
# 4. CHECK DATA FILE
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nCleaned dataset was not found:\n"
        f"{DATA_PATH}\n\n"
        f"Place your cleaned SEERA dataset there or "
        f"change DATA_PATH in this script."
    )


# ============================================================
# 5. LOAD CLEANED SEERA DATA
# ============================================================

print("=" * 75)
print("LOADING CLEANED SEERA DATA")
print("=" * 75)

df_work = pd.read_csv(
    DATA_PATH
)

print(
    f"Loaded shape: {df_work.shape}"
)


# ============================================================
# 6. STANDARDIZE COLUMN NAMES
# ============================================================

if isinstance(
    df_work.columns,
    pd.MultiIndex
):

    df_work.columns = [

        "_".join(
            str(level).strip()
            for level in col
            if str(level).strip() != ""
        )

        for col in df_work.columns
    ]


df_work.columns = (
    df_work.columns
    .astype(str)
    .str.replace(
        "\xa0",
        " ",
        regex=False
    )
    .str.strip()
)


# Remove duplicate columns
df_work = df_work.loc[
    :,
    ~df_work.columns.duplicated()
].copy()


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    target_col,
    group_col
]


for col in required_columns:

    if col not in df_work.columns:

        raise KeyError(
            f"Required column not found: {col}"
        )


# ============================================================
# 8. TARGET
# ============================================================

y = pd.to_numeric(
    df_work[target_col],
    errors="coerce"
)


valid_rows = y.notna()


df = (
    df_work
    .loc[valid_rows]
    .copy()
)


y = (
    y.loc[valid_rows]
    .reset_index(drop=True)
)


# ============================================================
# 9. FEATURES
# ============================================================

cols_to_drop = [

    target_col,

    group_col
]


if id_col in df.columns:

    cols_to_drop.append(
        id_col
    )


X = df.drop(
    columns=cols_to_drop
).copy()


# ============================================================
# 10. CATEGORICAL FEATURES
# ============================================================

nominal_cat_cols = [

    "Developers_Team selection",

    "Developers_DBMS expert availability",

    "General Information_Development type",

    "General Information_Application domain",

    "Environment_Government policy impact",

    "Environment_Developer hiring policy",

    "Environment_Developer training"
]


nominal_cat_cols = [

    col

    for col in nominal_cat_cols

    if col in X.columns
]


# CatBoost categorical representation
for col in nominal_cat_cols:

    X[col] = (

        X[col]

        .fillna("Missing")

        .astype(str)

        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )


# ============================================================
# 11. NUMERIC FEATURES
# ============================================================

numeric_cols = [

    col

    for col in X.columns

    if col not in nominal_cat_cols
]


for col in numeric_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# ============================================================
# 12. CATBOOST CATEGORY INDICES
# ============================================================

cat_indices = [

    X.columns.get_loc(col)

    for col in nominal_cat_cols
]


# ============================================================
# 13. TARGET TRANSFORMATION
# ============================================================

y_log = np.log1p(y)


# ============================================================
# 14. SAMPLE WEIGHTS
# ============================================================

sample_weights = np.where(

    y.to_numpy()
    < WEIGHT_THRESHOLD,

    LOW_EFFORT_WEIGHT,

    1.0
)


# ============================================================
# 15. DATA VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL DATA")
print("=" * 75)

print(
    f"Observations:          {len(X)}"
)

print(
    f"Features:              {X.shape[1]}"
)

print(
    f"Categorical features:  {len(cat_indices)}"
)

print(
    f"Organizations:         "
    f"{df[group_col].nunique()}"
)

print(
    f"Projects < 5,000h:     "
    f"{(y < WEIGHT_THRESHOLD).sum()}"
)

print(
    f"Projects >= 5,000h:    "
    f"{(y >= WEIGHT_THRESHOLD).sum()}"
)

print(
    f"Numerical NaNs:        "
    f"{X[numeric_cols].isna().sum().sum()}"
)


# ============================================================
# 16. TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL WEIGHTED CATBOOST MODEL")
print("=" * 75)


final_model = CatBoostRegressor(

    depth=FINAL_PARAMS["depth"],

    learning_rate=
        FINAL_PARAMS["learning_rate"],

    iterations=
        FINAL_PARAMS["iterations"],

    l2_leaf_reg=
        FINAL_PARAMS["l2_leaf_reg"],

    random_strength=
        FINAL_PARAMS["random_strength"],

    bagging_temperature=
        FINAL_PARAMS["bagging_temperature"],

    cat_features=cat_indices,

    loss_function="RMSE",

    random_seed=RANDOM_SEED,

    thread_count=-1,

    verbose=False,

    allow_writing_files=False
)


final_model.fit(

    X,

    y_log,

    sample_weight=sample_weights
)


print(
    "\nFinal model trained successfully."
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

final_model.save_model(
    MODEL_PATH
)


print(
    f"Model saved to:\n{MODEL_PATH}"
)


# ============================================================
# 18. SAVE METADATA
# ============================================================
#
# This allows the XAI script to reconstruct exactly the
# same feature representation.
# ============================================================

metadata = {

    "target_col":
        target_col,

    "group_col":
        group_col,

    "id_col":
        id_col,

    "feature_columns":
        X.columns.tolist(),

    "categorical_columns":
        nominal_cat_cols,

    "numeric_columns":
        numeric_cols,

    "categorical_indices":
        cat_indices,

    "weight_threshold":
        WEIGHT_THRESHOLD,

    "low_effort_weight":
        LOW_EFFORT_WEIGHT,

    "final_params":
        FINAL_PARAMS,

    "random_seed":
        RANDOM_SEED,

    "n_observations":
        int(len(X)),

    "n_features":
        int(X.shape[1])
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print(
    f"Metadata saved to:\n{METADATA_PATH}"
)


# ============================================================
# 19. DONE
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL CREATION COMPLETE")
print("=" * 75)

print(
    "\nSaved files:"
)

print(
    f"1. {MODEL_PATH}"
)

print(
    f"2. {METADATA_PATH}"
)