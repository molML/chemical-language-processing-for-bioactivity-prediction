import re
from typing import List

import selfies as sf
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)


def smiles_label_encoding(smiles: str, token_to_label: dict) -> List[int]:
    """Encode SMILES string to a list of integer

    Args:
        smiles (str): SMILES string
        token_to_label (dict): dictionary mapping token to integer

        Returns:
            List[int]: list of integer encoding the SMILES string
    """
    return [token_to_label[token] for token in segment_smiles(smiles)]


def selfies_label_encoding(selfies: str, token_to_label: dict) -> List[int]:
    """Encode SELFIES string to a list of integer

    Args:
        selfies (str): SELFIES string
        token_to_label (dict): dictionary mapping token to integer

        Returns:
            List[int]: list of integer encoding the SELFIES string
    """
    return [token_to_label[token] for token in sf.split_selfies(selfies)]
