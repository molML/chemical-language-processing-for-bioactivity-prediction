# %%
import json
import random

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#F7C59F", "#55B7FF", "#4F518C", "#55B7FFA0"]
PANEL_TO_COLOR = [
    ["#4F518C40", "#4F518C90", "#4F518CC0"],
    ["#4F518C40", "#4F518C90", "#4F518CC0"],
    ["#4F518C40", "#4F518C90", "#4F518CC0"],
    ["#4F518C40", "#4F518C90", "#4F518CC0", "#4F518CFF"],
]
DATASET_TO_NAME = {
    "DRD3": "DRD3\n(Class.)",
    "FEN1": "FEN1",
    "MAP4K2": "MAP4K2",
    "PIN1": "PIN1",
    "VDR": "VDR",
    "CHEMBL214_Ki": "5-HT1A",
    "CHEMBL233_Ki": "MOR",
    "CHEMBL234_Ki": "DRD3\n(Reg.)",
    "CHEMBL287_Ki": "SOR",
    "CHEMBL2147_Ki": "PIM1",
}
HP_TO_NAME = {
    "kernel_size": "Kernel Length",
    "n_filters": "Number of Filters",
    "n_layers": "Number of Convolution Layers",
    "embedding_dim": "Embedding Dimension",
}
representation = "smiles"
encoding = "learnable"
shuffling_scores = list()
dataset_to_counts = dict()
for DATABASE in ["excape", "mace"]:
    if DATABASE == "excape":
        model = "cnn_balanced"
        datasets = ["DRD3", "FEN1", "MAP4K2", "PIN1", "VDR"]
        metric = "balanced_accuracy"
    else:
        model = "cnn"
        datasets = [
            "CHEMBL214_Ki",
            "CHEMBL233_Ki",
            "CHEMBL234_Ki",
            "CHEMBL287_Ki",
            "CHEMBL2147_Ki",
        ]
        metric = "ci"

    for dataset in datasets:
        result_path = f"models/{DATABASE}/{dataset}/{representation}/{encoding}/{model}_results.json"
        with open(result_path, "r") as f:
            results = json.load(f)

        hp_scores = []
        n_hps = len(results)
        for combination_idx in range(n_hps):
            hp_results = results[f"combination-{combination_idx}"]
            kernel_size = hp_results["hps"]["kernel_size"]
            n_filters = hp_results["hps"]["n_filters"]
            n_layers = hp_results["hps"]["n_layers"]
            embedding_dim = hp_results["hps"]["embedding_dim"]
            score = hp_results["misc"][f"test_{metric}_mean"]
            hp_scores.append((kernel_size, n_filters, n_layers, embedding_dim, score))
        df_hp = pd.DataFrame(
            hp_scores,
            columns=["kernel_size", "n_filters", "n_layers", "embedding_dim", "score"],
        )
        df_hp = df_hp.sort_values(by="score", ascending=False)
        print("Dataset:", dataset)
        top_hps = df_hp.head(10)
        kernel_counts = top_hps["kernel_size"].value_counts().to_dict()
        filter_counts = top_hps["n_filters"].value_counts().to_dict()
        layer_counts = top_hps["n_layers"].value_counts().to_dict()
        dim_counts = top_hps["embedding_dim"].value_counts().to_dict()

        for value in [3, 5, 7]:
            if value not in kernel_counts:
                kernel_counts[value] = 0
        for value in [32, 64, 128]:
            if value not in filter_counts:
                filter_counts[value] = 0
        for value in [1, 2, 3]:
            if value not in layer_counts:
                layer_counts[value] = 0
        for value in [16, 32, 64, 128]:
            if value not in dim_counts:
                dim_counts[value] = 0

        dataset_to_counts[dataset] = {
            "kernel_size": kernel_counts,
            "n_filters": filter_counts,
            "n_layers": layer_counts,
            "embedding_dim": dim_counts,
        }

        scores = df_hp["score"].tolist()

        all_scores = []
        for idx in range(100):
            random.seed(idx)
            random.shuffle(scores)
            best_score_at_idx = [max(scores[: i + 1]) for i in range(len(scores))]
            all_scores.append(best_score_at_idx)

        all_scores = pd.DataFrame(all_scores)
        mean_scores = all_scores.mean(axis=0)
        std_scores = all_scores.std(axis=0)
        shuffling_scores.append((mean_scores, std_scores))


# %%%


hp_space = {
    "kernel_size": [3, 5, 7],
    "n_filters": [32, 64, 128],
    "n_layers": [1, 2, 3],
    "embedding_dim": [16, 32, 64, 128],
}
fig, grid = plt.subplots(3, 2, figsize=(8, 11))

dataset_names = list(DATASET_TO_NAME.keys())
for hp_idx, hp_name in enumerate(
    ["n_layers", "kernel_size", "n_filters", "embedding_dim"]
):
    bottom = [0] * len(dataset_names)
    ax = grid[hp_idx // 2, hp_idx % 2]
    colors = PANEL_TO_COLOR[hp_idx]
    for value_idx in range(4):
        if hp_name != "embedding_dim" and value_idx == 3:
            continue
        hp_value = hp_space[hp_name][value_idx]
        ax.barh(
            [DATASET_TO_NAME[dataset_name] for dataset_name in dataset_names],
            [
                dataset_to_counts[dataset_name][hp_name][hp_value]
                for dataset_name in dataset_names
            ],
            left=bottom,
            color=colors[value_idx],
            label=hp_value,
        )
        text_color = "black" if value_idx < 2 else "white"
        # text_color = "white"
        ax.bar_label(
            ax.containers[-1],
            labels=[
                dataset_to_counts[dataset_name][hp_name][hp_value]
                if dataset_to_counts[dataset_name][hp_name][hp_value] > 0
                else ""
                for dataset_name in dataset_names
            ],
            label_type="center",
            color=text_color,
        )
        bottom = [
            sum(x)
            for x in zip(
                bottom,
                [
                    dataset_to_counts[dataset_name][hp_name][hp_value]
                    for dataset_name in dataset_names
                ],
            )
        ]
        ax.legend(
            title=HP_TO_NAME[hp_name],
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.2),
            fontsize=8,
        )
        ax.text(
            -0.0,
            1.1,
            "abcd"[hp_idx],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks([])
    ax.set_xlim(0, 10)
    if hp_idx in [1, 3]:
        # move the ticks to the right
        ax.yaxis.tick_right()

    # if hp_idx in [2, 3]:
    #     ax.set_xlabel("Number of Models")


ax = grid[1, 1]
linecolors = [
    "#55B7FFFF",
    "#F7C59FFF",
    "#55B7FFA0",
    "#4F518CA0",
    "#4F518CFF",
    #### comment
    "#55B7FFFF",
    "#F7C59FFF",
    "#55B7FFA0",
    "#4F518CA0",
    "#4F518CFF",
]
dataset_idx = 0
for dataset_name, (mean_scores, std_scores) in zip(
    list(DATASET_TO_NAME.keys()), shuffling_scores
):
    # linestyle = "-" if dataset_idx < 5 else "--"
    ax = grid[2, 0] if dataset_idx < 5 else grid[2, 1]
    if "DRD3" in dataset_name or "CHEMBL234" in dataset_name:
        label = "DRD3"
    else:
        label = DATASET_TO_NAME[dataset_name]
    linestyle = "-"
    ax.plot(
        mean_scores,
        linestyle=linestyle,
        label=label,
        # color=linecolors[dataset_idx],
        linewidth=2,
    )
    ax.fill_between(
        mean_scores.index,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.3,
        # color=linecolors[dataset_idx],
    )
    ax.set_xlabel("Number of Models")
    ax.set_xticks([1, 100, 200, 300, 400])

    if dataset_idx == 4 or dataset_idx == 9:
        ax.vlines(
            len(df_hp) / 2,
            0.4,
            0.9,
            linestyle="--",
            color="black",
            alpha=0.2,
            # label="Half of the Models",
        )
        ax.legend(
            # title="ExCAPE-DB" if dataset_idx == 4 else "MoleculeACE",
            loc="upper center",
            # loc="lower right",
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            fontsize=8,
        )
        ax.text(
            -0.0,
            1.1,
            "ef"[dataset_idx // 5],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    if dataset_idx == 0:
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0.5, 0.86)

    if dataset_idx == 9:
        ax.yaxis.tick_right()
        # set right y label
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Concordance Index")
        ax.set_ylim(0.40, 0.76)

        # ax.set_yticks([])
    dataset_idx += 1


plt.tight_layout()
plt.savefig("./overleaf/figures/Figure5.pdf")
# %%
