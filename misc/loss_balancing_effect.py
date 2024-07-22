# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_best_hp_combination(results):
    n_hps = len(results)
    best_val_loss = float("inf")
    best_test_bacc, best_test_bacc_std, best_combination_idx = None, None, None
    for combination_idx in range(n_hps):
        hp_results = results[f"combination-{combination_idx}"]
        key = "val_loss_mean"
        if key not in hp_results["misc"]:
            key = "val_rmse_mean"
        combination_val_loss = hp_results["misc"][key]
        if combination_val_loss < best_val_loss:
            best_val_loss = combination_val_loss
            best_test_bacc = hp_results["misc"]["test_balanced_accuracy_mean"]
            best_test_bacc_std = hp_results["misc"]["test_balanced_accuracy_std"]
            best_combination_idx = combination_idx

    return best_test_bacc, best_test_bacc_std, best_combination_idx


def get_bests(df_results, groupby):
    df_bests = df_results.drop_duplicates(subset=["dataset", groupby], keep="first")
    df_bests = df_bests[
        ["dataset", groupby, f"test_{METRIC}_mean", f"test_{METRIC}_std"]
    ]
    return df_bests


REPRESENTATIONS = ["smiles", "selfies"]
ENCODINGS = ["learnable", "random", "onehot"]
MODELS = [
    "cnn",
    "gru",
    "transformer",
    "cnn_balanced",
    "gru_balanced",
    "transformer_balanced",
]
MODELS.sort()
DATASETS = [
    "DRD3",
    "FEN1",
    "MAP4K2",
    "PIN1",
    "VDR",
]
MODELS_TO_NAMES = {
    "cnn_balanced": "CNN",
    "gru_balanced": "RNN",
    "transformer_balanced": "Transformer",
    "xgboost_balanced": "XGBoost",
    "cnn": "CNN",
    "gru": "GRU",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
}
# NO_RW_COLOR = "#ff7f0e"
RW_COLOR = "#4F518C"


METRIC = "bacc"

clp_results = []
for dataset in DATASETS:
    for model in MODELS:
        for representation in REPRESENTATIONS:
            for encoding in ENCODINGS:
                result_path = f"models/excape/{dataset}/{representation}/{encoding}/{model}_results.json"
                if not os.path.exists(result_path):
                    continue
                with open(result_path, "r") as f:
                    results = json.load(f)

                test_score, test_score_std, best_combination = find_best_hp_combination(
                    results
                )

                clp_results.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "encoding": encoding,
                        "model": model,
                        "combination": best_combination,
                        f"test_{METRIC}_mean": test_score,
                        f"test_{METRIC}_std": test_score_std,
                    }
                )

xgb_results = []
for dataset in DATASETS:
    for xgb_type in ["xgboost", "xgboost_balanced"]:
        # for xgb_type in ["xgboost_balanced"]:
        result_path = f"models/excape/{dataset}/ecfp/{xgb_type}_results.json"
        with open(result_path, "r") as f:
            results = json.load(f)

        test_score, test_score_std, best_combination = find_best_hp_combination(results)
        xgb_results.append(
            {
                "dataset": dataset,
                "representation": "ecfp",
                "encoding": "ecfp",
                "model": xgb_type,
                "combination": best_combination,
                f"test_{METRIC}_mean": test_score,
                f"test_{METRIC}_std": test_score_std,
            }
        )


df_clp = pd.DataFrame(clp_results)
df_clp = df_clp.sort_values(
    by=["dataset", f"test_{METRIC}_mean"], ascending=[True, False]
)

df_xgb = pd.DataFrame(xgb_results)
df_xgb = df_xgb.sort_values(
    by=["dataset", f"test_{METRIC}_mean"], ascending=[True, False]
)
# %%
df_models = get_bests(df_clp, "model")
diffs = []
for dataset in DATASETS:
    df_dataset = df_models[df_models["dataset"] == dataset]
    for model in ["cnn", "gru", "transformer"]:
        df_model_no_bal = df_dataset[df_dataset["model"] == model]
        no_bal_score = df_model_no_bal[f"test_{METRIC}_mean"].values[0]

        df_model_bal = df_dataset[df_dataset["model"] == f"{model}_balanced"]
        bal_score = df_model_bal[f"test_{METRIC}_mean"].values[0]

        print(f"{dataset} {model}: {no_bal_score:.3f} vs {bal_score:.3f}")
        diffs.append(bal_score - no_bal_score)

print(f"Mean difference: {np.mean(diffs):.3f}")
# %%

fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.5, 2.5))
for i, dataset in enumerate(DATASETS):
    df_dataset = df_models[df_models["dataset"] == dataset]
    df_dataset = df_dataset.sort_values("model")
    balanced_models = df_dataset[df_dataset["model"].str.contains("_balanced")]
    unbalanced_models = df_dataset[~df_dataset["model"].str.contains("_balanced")]

    bar_width = 0.35
    index = np.arange(len(balanced_models))

    ax = axes[i]
    ax.margins(x=0.0)
    ax.bar(
        index,
        balanced_models[f"test_{METRIC}_mean"],
        bar_width,
        yerr=balanced_models[f"test_{METRIC}_std"],
        capsize=5,
        color=RW_COLOR,
        label="Loss re-weighting",
        zorder=1,
    )
    ax.bar(
        index + bar_width,
        unbalanced_models[f"test_{METRIC}_mean"],
        bar_width,
        yerr=unbalanced_models[f"test_{METRIC}_std"],
        capsize=5,
        # color=NO_RW_COLOR,
        color=RW_COLOR,
        alpha=0.35,
        label="No loss re-weighting",
        zorder=1,
    )

    if i == 0:
        ax.set_ylabel("Balanced Accuracy")
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    ax.set_ylim(0.5, 1)
    ax.set_title(dataset)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(
        [MODELS_TO_NAMES[model] for model in balanced_models["model"].tolist()]
    )

    xgb_balanced_score = df_xgb[df_xgb["dataset"] == dataset][
        f"test_{METRIC}_mean"
    ].values[0]
    xgb_balanced_std = df_xgb[df_xgb["dataset"] == dataset][
        f"test_{METRIC}_std"
    ].values[0]
    ax.hlines(
        xgb_balanced_score,
        -0.35,
        2.65,
        color=RW_COLOR,
        linestyle="--",
        zorder=0,
    )
    ax.fill_between(
        [-0.35, 2.65],
        xgb_balanced_score - xgb_balanced_std,
        xgb_balanced_score + xgb_balanced_std,
        color=RW_COLOR,
        alpha=0.1,
        zorder=0,
    )

    xgb_unbalanced_score = df_xgb[df_xgb["dataset"] == dataset][
        f"test_{METRIC}_mean"
    ].values[1]
    xgb_unbalanced_std = df_xgb[df_xgb["dataset"] == dataset][
        f"test_{METRIC}_std"
    ].values[1]
    ax.hlines(
        xgb_unbalanced_score,
        -0.35,
        2.65,
        color=RW_COLOR,
        alpha=0.35,
        linestyle="--",
        zorder=0,
    )
    ax.fill_between(
        [-0.35, 2.65],
        xgb_unbalanced_score - xgb_unbalanced_std,
        xgb_unbalanced_score + xgb_unbalanced_std,
        color=RW_COLOR,
        alpha=0.1,
        zorder=0,
    )


plt.tight_layout()
# plt.legend(ncol=2, loc=(-2.35, 1.2))
plt.legend(ncol=2, loc="center", bbox_to_anchor=(-1.75, 1.3))
plt.savefig(
    "./hitchhiker-guide-to-bioactivity-pred/figures/excape/loss_reweighting.pdf",
    bbox_inches="tight",
)
plt.show()

# %%
