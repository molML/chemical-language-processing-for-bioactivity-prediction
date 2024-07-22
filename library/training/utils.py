import re
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

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


def segment_smiles_batch(
    smiles_batch: List[str], segment_sq_brackets=True
) -> List[List[str]]:
    return [segment_smiles(smiles, segment_sq_brackets) for smiles in smiles_batch]


def clean_smiles(
    smiles: str,
    remove_salt=True,
    desalt=False,
    uncharge=True,
    sanitize=True,
    remove_stereochemistry=True,
    to_canonical=True,
):
    if remove_salt and is_salt(smiles):
        return None

    if remove_stereochemistry:
        smiles = drop_stereochemistry(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    salt_remover = SaltRemover()
    uncharger = rdMolStandardize.Uncharger()
    if desalt:
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if uncharge:
        mol = uncharger.uncharge(mol)
    if sanitize:
        sanitization_flag = Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
        )
        # SANITIZE_NONE is the "no error" flag of rdkit!
        if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None

    return Chem.MolToSmiles(mol, canonical=to_canonical)


def clean_smiles_batch(
    smiles_batch: List[str],
    remove_salt=True,
    desalt=False,
    uncharge=True,
    sanitize=True,
    remove_stereochemistry=True,
    to_canonical=True,
):
    cleaned = [
        clean_smiles(
            smiles,
            remove_salt=remove_salt,
            desalt=desalt,
            uncharge=uncharge,
            sanitize=sanitize,
            remove_stereochemistry=remove_stereochemistry,
            to_canonical=to_canonical,
        )
        for smiles in smiles_batch
    ]
    return [smiles for smiles in cleaned if smiles is not None]


def is_salt(smiles: str, negate_result=False) -> bool:
    is_salt = "." in set(smiles)
    if negate_result:
        return not is_salt
    return is_salt


def drop_stereochemistry(smiles: str):
    replace = {ord("/"): None, ord("\\"): None, ord("@"): None}
    return smiles.translate(replace)
