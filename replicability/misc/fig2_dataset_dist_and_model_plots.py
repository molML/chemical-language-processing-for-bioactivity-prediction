# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def compute_max_tanimoto(train_smiles, test_smiles):
    train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
    test_mols = [Chem.MolFromSmiles(smiles) for smiles in test_smiles]

    # Convert molecules to fingerprints
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols
    ]
    test_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in test_mols
    ]

    max_sim_test = [
        max(DataStructs.BulkTanimotoSimilarity(test_fp, train_fps))
        for test_fp in test_fps
    ]

    return max_sim_test


def compute_train_test_sims(datasets):
    all_sims = list()
    for dataset_idx in range(len(datasets)):
        dataset = datasets[dataset_idx]
        print(f"Dataset: {dataset}")
        dataset_train_test_sims = list()
        for setup_ix in range(5):
            df_train = pd.read_csv(f"./data/{dataset}/setup_{setup_ix}/train.csv")
            train_smiles = df_train["cleaned_smiles"].tolist()

            df_test = pd.read_csv(f"./data/{dataset}/setup_{setup_ix}/test.csv")
            test_smiles = df_test["cleaned_smiles"].tolist()

            test_sim = compute_max_tanimoto(train_smiles, test_smiles)
            dataset_train_test_sims.extend(test_sim)

        all_sims.append(dataset_train_test_sims)

    return all_sims


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

# excape_sims = compute_train_test_sims(excape_datasets)
# mace_sims = compute_train_test_sims(mace_datasets)
# excape_distances = (1 - np.array(excape_sims)).tolist()
# mace_distances = [(1 - np.array(mace_sim)).tolist() for mace_sim in mace_sims]

# with open("./results/excape_distances.txt", "w") as f:
#     f.write(str(excape_distances))

# with open("./results/mace_distances.txt", "w") as f:
#     f.write(str(mace_distances))

with open("./results/excape_distances.txt", "r") as f:
    excape_distances = eval(f.read())

with open("./results/mace_distances.txt", "r") as f:
    mace_distances = eval(f.read())

excape_sims = [[1 - dist for dist in lst] for lst in excape_distances]
mace_sims = [[1 - dist for dist in lst] for lst in mace_distances]
df_clp_excape = pd.read_csv("./results/excape_clp_results_bacc_models.csv")
df_clp_excape = df_clp_excape.replace(
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
df_clp_mace = pd.read_csv("./results/mace_clp_results_ci_models.csv")
df_clp_mace = df_clp_mace.replace(
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

# %%
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

TRAIN_TEST_DIST_COLOR = "#4F518C"

MODELS_TO_NAMES = {
    "cnn": "CNN",
    "gru": "RNN",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
}
MODELS_TO_COLORS = {
    "cnn": "#F7C59F",
    "gru": "#55B7FF",
    "transformer": "#4F518C",
    # "gru": "#2176AE",
    # "transformer": "#0B6E4F",
    # "transformer": "#0000FF80",
    # "xgboost": "#2176AE",
    "xgboost": "#8b9daf",
}

METRIC_TO_NAME = {
    "bacc": "Balanced Accuracy",
    "ci": "Concordance Index",
}


def plot_dataset_distance(axes, distances, datasets, database_name, letter_idx, y_pos):
    for column_idx in range(5):
        sns.kdeplot(
            distances[column_idx],
            color=TRAIN_TEST_DIST_COLOR,
            ax=axes[column_idx],
            fill=True,
            # cumulative=True,
        )
        axes[column_idx].set_title(DATASET_TO_NAME[datasets[column_idx]])
        axes[column_idx].set_xlabel("Test Set Similarity")
        axes[column_idx].set_ylim(0, 12)
        axes[column_idx].set_xlim(0, 1)
        axes[column_idx].set_xticks([0, 0.5, 1])
        if column_idx != 0:
            axes[column_idx].set_ylabel("")
            axes[column_idx].set_yticks([])
        else:
            axes[column_idx].set_ylabel(f"{database_name}\nMolecule Density")

    fig.text(0.01, y_pos, letter_idx, fontdict={"size": 14, "weight": "bold"})


def plot_model_performance(
    df_clp, axes, datasets, metric, df_xgb, database_name, letter_idx, y_pos
):
    for dataset_idx, dataset in enumerate(datasets):
        df_dataset = df_clp[df_clp["dataset"] == dataset]
        df_dataset = df_dataset.sort_values("model")
        ax = axes[dataset_idx]
        ax.margins(x=0.0)
        ax.bar(
            df_dataset["model"],
            df_dataset[f"test_{metric}_mean"],
            yerr=df_dataset[f"test_{metric}_std"],
            capsize=5,
            color=[MODELS_TO_COLORS[model] for model in df_dataset["model"]],
            zorder=1,
        )
        ax.hlines(
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0],
            -0.65,
            2.65,
            color=MODELS_TO_COLORS["xgboost"],
            linestyle="--",
            zorder=0,
        )
        ax.fill_between(
            [-0.65, 2.65],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            - df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_mean"].values[0]
            + df_xgb[df_xgb["dataset"] == dataset][f"test_{metric}_std"].values[0],
            color=MODELS_TO_COLORS["xgboost"],
            alpha=0.1,
            zorder=0,
        )
        ax.set_title(DATASET_TO_NAME[dataset])
        ax.set_xticklabels(
            [MODELS_TO_NAMES[model] for model in df_dataset["model"].tolist()]
        )
        ax.set_ylabel(f"{database_name}\n{METRIC_TO_NAME[metric]}")
        ax.set_ylim(0.5, 1)

        # ax.set_xlabel("Architecture")

        if dataset_idx != 0:
            ax.set_ylabel("")
            ax.set_yticks([])

    fig.text(0.01, y_pos, letter_idx, fontdict={"size": 14, "weight": "bold"})


fig, axes = plt.subplots(4, 5, figsize=(12.5, 10))
plot_dataset_distance(
    axes[0], excape_sims, excape_datasets, "Classification", "a", 0.975
)
plot_dataset_distance(axes[1], mace_sims, mace_datasets, "Regression", "b", 0.725)

# plt.subplots_adjust(left=0.0, right=1.0, hspace=0.0)
plot_model_performance(
    df_clp_excape,
    axes[2],
    excape_datasets,
    "bacc",
    df_xgb_excape,
    "Classification",
    "c",
    0.475,
)
plot_model_performance(
    df_clp_mace, axes[3], mace_datasets, "ci", df_xgb_mace, "Regression", "d", 0.225
)


plt.tight_layout()
plt.savefig("./overleaf/figures/Figure2.pdf")
plt.show()

# %%
