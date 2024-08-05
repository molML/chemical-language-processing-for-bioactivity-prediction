# %%
import json

import pandas as pd
import selfies as sf

from emb_library.training.utils import clean_smiles_batch, segment_smiles_batch


def label_encoding(tokenized_inputs, token2label):
    return [[token2label[token] for token in inp] for inp in tokenized_inputs]


def smiles_to_selfies_encoding(smiles, token2label):
    sfs = [sf.encoder(s) for s in smiles]
    lens = [sf.len_selfies(s) for s in sfs]
    if max(lens) > 80:
        print("max len:", max(lens))

    tokenized = [sf.split_selfies(s) for s in sfs]
    return [[token2label[s] for s in t] for t in tokenized]


MAX_SMILES_LEN = 0

with open("./data/token2label_smiles.json") as f:
    smiles_token2label = json.load(f)

with open("./data/token2label_selfies_cls.json") as f:
    selfies_token2label = json.load(f)

datasets = ["DRD3", "FEN1", "MAP4K2", "PIN1", "VDR"]

for protein_name in datasets:
    for setup_idx in range(5):
        print(protein_name, setup_idx)
        for fold in ["train", "valid", "test"]:
            with open(
                f"./data/distant/{protein_name}/setup_{setup_idx}/{fold}.smiles"
            ) as f:
                lines = f.readlines()
                raw_smiles = [line.strip().split(",")[0] for line in lines]
                y = [int(line.strip().split(",")[1]) for line in lines]

            cleaned_smiles = clean_smiles_batch(raw_smiles)
            if len(cleaned_smiles) != len(raw_smiles):
                raise ValueError("Some mols are lost!")
            segmented_smiles = segment_smiles_batch(
                cleaned_smiles, segment_sq_brackets=True
            )
            sanity = ["".join(s) for s in segmented_smiles]
            if len(set(sanity)) != len(sanity):
                raise ValueError("Duplicate SMILES!")
            if sanity != cleaned_smiles:
                raise ValueError("Segmentation failed!")
            if len(sanity) != len(set(sanity)):
                raise ValueError("Duplicate SMILES!")

            encoded_smiles = label_encoding(segmented_smiles, smiles_token2label)
            lens = [len(s) for s in encoded_smiles]
            max_len = max(lens)
            if max_len > MAX_SMILES_LEN:
                MAX_SMILES_LEN = max_len
                print(MAX_SMILES_LEN)

            encoded_selfies = smiles_to_selfies_encoding(
                cleaned_smiles, selfies_token2label
            )
            df = pd.DataFrame(
                {
                    "cleaned_smiles": cleaned_smiles,
                    "encoded_smiles": encoded_smiles,
                    "encoded_selfies": encoded_selfies,
                    "y": y,
                }
            )
            df.to_csv(
                f"./data/distant/{protein_name}/setup_{setup_idx}/{fold}.csv",
                index=False,
            )
