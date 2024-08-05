# %%
import matplotlib.pyplot as plt
import pandas as pd

df_clp_excape_reprs = pd.read_csv(
    "./results/excape_clp_results_bacc_representations.csv"
)
df_clp_excape_reprs = df_clp_excape_reprs.replace(
    {
        "dataset": {
            "DRD3": "excape/DRD3",
            "FEN1": "excape/FEN1",
            "MAP4K2": "excape/MAP4K2",
            "PIN1": "excape/PIN1",
            "VDR": "excape/VDR",
        },
        "model": {
            "cnn_balanced": "cnn",
            "gru_balanced": "gru",
            "transformer_balanced": "transformer",
            "xgboost_balanced": "xgboost",
        },
    }
)
df_clp_excape_encodings = pd.read_csv("./results/excape_clp_results_bacc_encodings.csv")
df_clp_excape_encodings = df_clp_excape_encodings.replace(
    {
        "dataset": {
            "DRD3": "excape/DRD3",
            "FEN1": "excape/FEN1",
            "MAP4K2": "excape/MAP4K2",
            "PIN1": "excape/PIN1",
            "VDR": "excape/VDR",
        },
        "model": {
            "cnn_balanced": "cnn",
            "gru_balanced": "gru",
            "transformer_balanced": "transformer",
            "xgboost_balanced": "xgboost",
        },
    }
)

df_clp_mace_reprs = pd.read_csv("./results/mace_clp_results_ci_representations.csv")
df_clp_mace_reprs = df_clp_mace_reprs.replace(
    {
        "dataset": {
            "CHEMBL214_Ki": "mace/CHEMBL214_Ki",
            "CHEMBL233_Ki": "mace/CHEMBL233_Ki",
            "CHEMBL234_Ki": "mace/CHEMBL234_Ki",
            "CHEMBL287_Ki": "mace/CHEMBL287_Ki",
            "CHEMBL2147_Ki": "mace/CHEMBL2147_Ki",
        }
    }
)
df_clp_mace_encodings = pd.read_csv("./results/mace_clp_results_ci_encodings.csv")
df_clp_mace_encodings = df_clp_mace_encodings.replace(
    {
        "dataset": {
            "CHEMBL214_Ki": "mace/CHEMBL214_Ki",
            "CHEMBL233_Ki": "mace/CHEMBL233_Ki",
            "CHEMBL234_Ki": "mace/CHEMBL234_Ki",
            "CHEMBL287_Ki": "mace/CHEMBL287_Ki",
            "CHEMBL2147_Ki": "mace/CHEMBL2147_Ki",
        }
    }
)


df_xgb_excape = pd.read_csv("./results/excape_xgb_results_bacc.csv")
df_xgb_excape = df_xgb_excape.replace(
    {
        "dataset": {
            "DRD3": "excape/DRD3",
            "FEN1": "excape/FEN1",
            "MAP4K2": "excape/MAP4K2",
            "PIN1": "excape/PIN1",
            "VDR": "excape/VDR",
        }
    }
)

df_xgb_mace = pd.read_csv("./results/mace_xgb_results_ci.csv")
df_xgb_mace = df_xgb_mace.replace(
    {
        "dataset": {
            "CHEMBL214_Ki": "mace/CHEMBL214_Ki",
            "CHEMBL233_Ki": "mace/CHEMBL233_Ki",
            "CHEMBL234_Ki": "mace/CHEMBL234_Ki",
            "CHEMBL287_Ki": "mace/CHEMBL287_Ki",
            "CHEMBL2147_Ki": "mace/CHEMBL2147_Ki",
        }
    }
)


excape_datasets = [
    "excape/DRD3",
    "excape/FEN1",
    "excape/MAP4K2",
    "excape/PIN1",
    "excape/VDR",
]
mace_datasets = [
    "mace/CHEMBL214_Ki",
    "mace/CHEMBL233_Ki",
    "mace/CHEMBL234_Ki",
    "mace/CHEMBL287_Ki",
    "mace/CHEMBL2147_Ki",
]
DATASET_TO_NAME = {
    "excape/DRD3": "DRD3",
    "excape/FEN1": "FEN1",
    "excape/MAP4K2": "MAP4K2",
    "excape/PIN1": "PIN1",
    "excape/VDR": "VDR",
    "mace/CHEMBL214_Ki": "5-HT1A",
    "mace/CHEMBL233_Ki": "MOR",
    "mace/CHEMBL234_Ki": "DRD3",
    "mace/CHEMBL287_Ki": "SOR",
    "mace/CHEMBL2147_Ki": "PIM1",
}

REPRESENTATIONS_TO_COLORS = {
    "smiles": "#55B7FF",
    "selfies": "#F7C59F",
}
ENCODINGS_TO_COLORS = {
    "onehot": "#F7C59F",
    "learnable": "#4F518C",
    "random": "#55B7FF",
}

METRIC_TO_NAME = {
    "bacc": "Balanced Accuracy",
    "ci": "Concordance Index",
}
XGB_COLOR = "#8D9CAF"


def plot_representations(
    df_reprs, datasets, metric, df_xgb, axes, y_pos, letter_idx, database_name
):
    for i, dataset in enumerate(datasets):
        df_dataset = df_reprs[df_reprs["dataset"] == dataset]
        df_dataset = df_dataset.sort_values("representation")
        ax = axes[i]
        ax.margins(x=0)
        ax.bar(
            df_dataset["representation"],
            df_dataset[f"test_{metric}_mean"],
            yerr=df_dataset[f"test_{metric}_std"],
            capsize=5,
            color=[
                REPRESENTATIONS_TO_COLORS[representation]
                for representation in df_dataset["representation"]
            ],
            zorder=1,
        )
        ax.set_title(DATASET_TO_NAME[dataset])
        ax.set_ylabel(f"{database_name}\n{METRIC_TO_NAME[metric]}")
        ax.set_ylim(0.5, 1)
        ax.set_xticklabels(
            [
                representation.upper()
                for representation in df_dataset["representation"].tolist()
            ]
        )

        ax.hlines(
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0],
            -0.65,
            len(REPRESENTATIONS_TO_COLORS) - 0.35,
            color=XGB_COLOR,
            linestyle="--",
            zorder=0,
        )
        ax.fill_between(
            [-0.65, len(REPRESENTATIONS_TO_COLORS) - 0.35],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            - df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            + df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            color=XGB_COLOR,
            alpha=0.1,
            zorder=0,
        )

        if i != 0:
            ax.set_ylabel("")
            ax.set_yticks([])

    fig.text(0.01, y_pos, letter_idx, fontdict={"size": 14, "weight": "bold"})


def plot_encodings(
    df_encodings, datasets, metric, df_xgb, axes, y_pos, letter_idx, database_name
):
    for i, dataset in enumerate(datasets):
        df_dataset = df_encodings[df_encodings["dataset"] == dataset]
        df_dataset = df_dataset.sort_values("encoding")
        df_dataset["order"] = [2, 0, 1]
        df_dataset = df_dataset.sort_values("order")
        ax = axes[i]
        ax.margins(x=0)
        ax.bar(
            df_dataset["encoding"],
            df_dataset[f"test_{metric}_mean"],
            yerr=df_dataset[f"test_{metric}_std"],
            capsize=5,
            color=[
                ENCODINGS_TO_COLORS[encoding] for encoding in df_dataset["encoding"]
            ],
            zorder=1,
        )
        ax.set_title(DATASET_TO_NAME[dataset])
        ax.set_ylabel(f"{database_name}\n{METRIC_TO_NAME[metric]}")
        ax.set_ylim(0.5, 1)
        ax.set_xticklabels(
            [encoding.title() for encoding in df_dataset["encoding"].tolist()]
        )

        ax.hlines(
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0],
            -0.65,
            len(ENCODINGS_TO_COLORS) - 0.35,
            color=XGB_COLOR,
            linestyle="--",
            zorder=0,
        )
        ax.fill_between(
            [-0.65, len(ENCODINGS_TO_COLORS) - 0.35],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            - df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            + df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            color=XGB_COLOR,
            alpha=0.1,
            zorder=0,
        )

        if i != 0:
            ax.set_ylabel("")
            ax.set_yticks([])

    fig.text(0.01, y_pos, letter_idx, fontdict={"size": 14, "weight": "bold"})


fig, axes = plt.subplots(4, 5, figsize=(12.5, 10))

plot_representations(
    df_clp_excape_reprs,
    excape_datasets,
    "bacc",
    df_xgb_excape,
    axes[0],
    0.975,
    "a",
    "Classification",
)
plot_representations(
    df_clp_mace_reprs,
    mace_datasets,
    "ci",
    df_xgb_mace,
    axes[1],
    0.725,
    "b",
    "Regression",
)

plot_encodings(
    df_clp_excape_encodings,
    excape_datasets,
    "bacc",
    df_xgb_excape,
    axes[2],
    0.475,
    "c",
    "Classification",
)
plot_encodings(
    df_clp_mace_encodings,
    mace_datasets,
    "ci",
    df_xgb_mace,
    axes[3],
    0.225,
    "d",
    "Regression",
)


plt.tight_layout()
plt.savefig(
    "./hitchhiker-guide-to-bioactivity-pred/figures/representations_and_encodings.pdf"
)
plt.show()

# %%
