import itertools
import json

__HP_MODELS = {
    "cnn": {
        "n_layers": [1, 2, 3],
        "kernel_size": [3, 5, 7],
        "n_filters": [32, 64, 128],
        "dense_layer_size": [64],
        "dropout": [0.25],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "batch_size": [32],
    },
    "transformer": {
        "n_layers": [1, 2, 3],
        "n_heads": [1, 2, 4],
        "ff_dim": [32, 64, 128],
        "dropout": [0.25],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
        "batch_size": [32],
    },
    "gru": {
        "n_layers": [1, 2, 3],
        "gru_dim": [16, 32, 64, 128],
        "dense_layer_size": [64],
        "dropout": [0.25],
        "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "batch_size": [32],
    },
    "xgboost": {
        "n_estimators": [2000],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "colsample_bytree": [0.5, 0.75, 1.0],
        "subsample": [0, 0.5, 0.75, 1.0],
        "random_state": [42],
        "n_jobs": [-1],
    },
}

__HP_EMBEDDINGS = {
    "random": {
        "embedding_dim": [16, 32, 64, 128],
    },
    "onehot": {},
    "learnable": {
        "embedding_dim": [16, 32, 64, 128],
    },
}

__HP_REPRESENTATIONS = {
    "smiles": {
        "maxlen": [85],
        "vocab_size": [35],
    },
    "selfies": {
        "maxlen": [85],
        "vocab_size": [50],
    },
}


def create_hp_space(representation_name: str, embedding_name: str, model_name: str):
    representation_hp_space = __HP_REPRESENTATIONS[representation_name]
    embedding_hp_space = __HP_EMBEDDINGS[embedding_name]
    model_hp_space = __HP_MODELS[model_name]

    merged_space = {
        **representation_hp_space,
        **embedding_hp_space,
        **model_hp_space,
    }
    combinations = list(itertools.product(*merged_space.values()))
    sampled_space = [
        dict(zip(merged_space.keys(), combination)) for combination in combinations
    ]
    idxed_sampled_space = [
        {"combination_idx": idx, **combination}
        for idx, combination in enumerate(sampled_space)
    ]
    print(f"Created {len(idxed_sampled_space)} combinations")
    return idxed_sampled_space


def get_sampled_hp_space(representation_name: str, encoding_name: str, model_name: str):
    with open(
        f"./library/training/spaces/{representation_name}/{encoding_name}/{model_name}.json",
        "r",
    ) as f:
        sampled_space = json.load(f)
    return sampled_space


if __name__ == "__main__":
    import os

    for representation_name in __HP_REPRESENTATIONS.keys():
        for embedding_name in __HP_EMBEDDINGS.keys():
            for model_name in __HP_MODELS.keys():
                print(representation_name, embedding_name, model_name)
                sampled_space = create_hp_space(
                    representation_name, embedding_name, model_name
                )
                savedir = (
                    f"./library/training/spaces/{representation_name}/{embedding_name}"
                )
                os.makedirs(savedir, exist_ok=True)
                with open(f"./{savedir}/{model_name}.json", "w") as f:
                    json.dump(sampled_space, f, indent=4)
