# ============================================================
# xai_analysis.py
#
# FINAL WEIGHTED CATBOOST — XAI
#
# Loads the frozen model created by train_final_model.py
#
# Explanation methods:
#   1. SHAP global feature importance
#   2. SHAP global beeswarm
#   3. SHAP local waterfall
#   4. DiCE counterfactual explanation
#
# IMPORTANT:
#   This script does NOT evaluate predictive performance.
#
#   The model has already been evaluated using 41-fold LOGO.
#   Here it is loaded for interpretation only.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import dice_ml

from catboost import CatBoostRegressor


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# xai_analysis.py:
#
# Project practical/
#     xai/
#         xai_analysis.py
#
# parents[1] = Project practical/
# ============================================================
# PROJECT PATHS
# ============================================================

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "seera_cleaned.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
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
OUTPUT_DIR = (
    PROJECT_ROOT
    /"results"
    /"xai"
)

# ============================================================
# 3. CHECK FILES
# ============================================================

required_files = [

    DATA_PATH,

    MODEL_PATH,

    METADATA_PATH
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path}"
        )


# ============================================================
# 4. LOAD METADATA
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


target_col = metadata[
    "target_col"
]

group_col = metadata[
    "group_col"
]

id_col = metadata[
    "id_col"
]

feature_columns = metadata[
    "feature_columns"
]

nominal_cat_cols = metadata[
    "categorical_columns"
]

numeric_cols = metadata[
    "numeric_columns"
]

cat_indices = metadata[
    "categorical_indices"
]

weight_threshold = metadata[
    "weight_threshold"
]


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 75)
print("LOADING DATA AND FINAL MODEL")
print("=" * 75)


df_work = pd.read_csv(
    DATA_PATH
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
            if str(level).strip()
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


df_work = df_work.loc[
    :,
    ~df_work.columns.duplicated()
].copy()


# ============================================================
# 7. TARGET
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
# 8. REBUILD FEATURE MATRIX
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
# 9. VERIFY FEATURE ALIGNMENT
# ============================================================

if X.columns.tolist() != feature_columns:

    raise ValueError(
        "\nFeature mismatch detected.\n\n"

        "Features used when the model was trained "
        "do not match the current dataset.\n\n"

        f"Expected:\n{feature_columns}\n\n"

        f"Found:\n{X.columns.tolist()}"
    )


# ============================================================
# 10. CATEGORICAL FEATURES
# ============================================================

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
# 11. NUMERICAL FEATURES
# ============================================================

for col in numeric_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# ============================================================
# 12. LOAD FINAL CATBOOST MODEL
# ============================================================

final_model = CatBoostRegressor()

final_model.load_model(
    MODEL_PATH
)


print(
    f"Model loaded from:\n{MODEL_PATH}"
)

print(
    f"Observations: {len(X)}"
)

print(
    f"Features:     {X.shape[1]}"
)


# ============================================================
# 13. CREATE OUTPUT DIRECTORY
# ============================================================

print(
    f"\nXAI results will be saved to:\n"
    f"{OUTPUT_DIR}"
)


# ============================================================
# 14. SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 75)
print("CREATING SHAP EXPLANATIONS")
print("=" * 75)


explainer = shap.TreeExplainer(
    final_model
)


shap_values = explainer(
    X
)


# ============================================================
# 15. SHAP GLOBAL BAR PLOT
# ============================================================

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "shap_global_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: shap_global_bar.png"
)


# ============================================================
# 16. SHAP GLOBAL BEESWARM
# ============================================================

plt.figure(
    figsize=(10, 7)
)

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "shap_global_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: shap_global_beeswarm.png"
)


# ============================================================
# 17. GLOBAL SHAP IMPORTANCE TABLE
# ============================================================

global_importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Mean_Absolute_SHAP":
        np.abs(
            shap_values.values
        ).mean(axis=0)
})


global_importance = (
    global_importance
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


global_importance.to_csv(
    OUTPUT_DIR
    / "shap_global_importance.csv",
    index=False
)


print(
    "Saved: shap_global_importance.csv"
)


# ============================================================
# 18. LOCAL SHAP WATERFALL
# ============================================================
#
# Change PROJECT_INDEX to explain another project.
# ============================================================

PROJECT_INDEX = 0


if not (
    0
    <= PROJECT_INDEX
    < len(X)
):

    raise IndexError(
        f"PROJECT_INDEX must be between "
        f"0 and {len(X)-1}."
    )


query_instance = X.iloc[
    [PROJECT_INDEX]
]


predicted_log = (
    final_model.predict(
        query_instance
    )[0]
)


predicted_effort = np.expm1(
    predicted_log
)


actual_effort = y.iloc[
    PROJECT_INDEX
]


print("\n" + "=" * 75)
print("LOCAL SHAP EXPLANATION")
print("=" * 75)

print(
    f"Project index:    {PROJECT_INDEX}"
)

print(
    f"Actual effort:    "
    f"{actual_effort:.2f} hours"
)

print(
    f"Predicted effort: "
    f"{predicted_effort:.2f} hours"
)


plt.figure(
    figsize=(10, 7)
)

shap.plots.waterfall(
    shap_values[
        PROJECT_INDEX
    ],
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "shap_local_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved: shap_local_waterfall.png"
)


# ============================================================
# 19. LOCAL SHAP TABLE
# ============================================================

local_shap = pd.DataFrame({

    "Feature":
        X.columns,

    "Feature_Value":
        X.iloc[
            PROJECT_INDEX
        ].values,

    "SHAP_Value":
        shap_values[
            PROJECT_INDEX
        ].values
})


local_shap["Absolute_SHAP"] = (
    local_shap["SHAP_Value"]
    .abs()
)


local_shap = (
    local_shap
    .sort_values(
        "Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


local_shap.to_csv(
    OUTPUT_DIR
    / "shap_local_values.csv",
    index=False
)


print(
    "Saved: shap_local_values.csv"
)


# ============================================================
# 20. PREPARE DATA FOR DiCE
# ============================================================
#
# DiCE works with the original target scale.
#
# The CatBoost model itself predicts log1p(effort), so
# the wrapper below converts its output back to hours.
# ============================================================

df_dice = X.copy()

df_dice[
    "Effort_Actual_effort"
] = y.to_numpy()


categorical_features = (
    nominal_cat_cols.copy()
)


continuous_features = [

    col

    for col in X.columns

    if col not in categorical_features
]


# ============================================================
# 21. CREATE DiCE DATA OBJECT
# ============================================================

d = dice_ml.Data(

    dataframe=df_dice,

    continuous_features=
        continuous_features,

    categorical_features=
        categorical_features,

    outcome_name=
        "Effort_Actual_effort"
)


# ============================================================
# 22. CATBOOST → DiCE WRAPPER
# ============================================================

class CatBoostDiCEWrapper:

    def __init__(
        self,
        model
    ):

        self.model = model


    def predict(
        self,
        df_input
    ):

        pred_log = (
            self.model.predict(
                df_input
            )
        )

        return np.clip(
            np.expm1(pred_log),
            a_min=0,
            a_max=None
        )


dice_model = (
    CatBoostDiCEWrapper(
        final_model
    )
)


# ============================================================
# 23. CREATE DiCE EXPLAINER
# ============================================================

m = dice_ml.Model(

    model=dice_model,

    backend="sklearn",

    model_type="regressor"
)


exp = dice_ml.Dice(

    d,

    m,

    method="random"
)


# ============================================================
# 24. CURRENT PROJECT
# ============================================================

current_prediction = np.expm1(

    final_model.predict(
        query_instance
    )[0]
)


print("\n" + "=" * 75)
print("DiCE COUNTERFACTUAL EXPLANATION")
print("=" * 75)

print(
    f"Project index:       {PROJECT_INDEX}"
)

print(
    f"Actual effort:       "
    f"{actual_effort:.2f} hours"
)

print(
    f"Current prediction:  "
    f"{current_prediction:.2f} hours"
)


# ============================================================
# 25. DESIRED COUNTERFACTUAL RANGE
# ============================================================
#
# This asks for alternatives whose predicted effort is
# between 50% and 80% of the current prediction.
#
# This is a demonstration scenario, not a causal claim.
# ============================================================

target_range = [

    current_prediction * 0.50,

    current_prediction * 0.80
]


print(
    f"Desired predicted range: "
    f"{target_range[0]:.2f} - "
    f"{target_range[1]:.2f} hours"
)


# ============================================================
# 26. GENERATE COUNTERFACTUALS
# ============================================================

dice_exp = exp.generate_counterfactuals(

    query_instance,

    total_CFs=2,

    desired_range=target_range
)


# ============================================================
# 27. DISPLAY COUNTERFACTUALS
# ============================================================

cf_df = (
    dice_exp
    .cf_examples_list[0]
    .final_cfs_df
)


print("\nCounterfactual explanations:")

print(
    cf_df.to_string(
        index=False
    )
)


# ============================================================
# 28. SAVE COUNTERFACTUALS
# ============================================================

cf_df.to_csv(
    OUTPUT_DIR
    / "dice_counterfactuals.csv",
    index=False
)


print(
    "\nSaved: dice_counterfactuals.csv"
)


# ============================================================
# 29. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("XAI ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"\nResults directory:\n{OUTPUT_DIR}"
)