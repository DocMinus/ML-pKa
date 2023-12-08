#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Version 0.5.7 (May 24, 07:30:00 2021)
Update: 2023-01-30
@author: Alexander Minidis (DocMinus)
Usage:
from mymodules import chemtools (for all)
or
from mymodules.chemtools import modulename
Copyright (c) 2021-2022 DocMinus
"""

"""
Changelog: 
added a read A+B->C+D reactions function
"""

import re
import time
import pandas as pd

# RDkit stuff
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import RDLogger

# Important, or else waaaay too many RDkit details in output
RDLogger.logger().setLevel(RDLogger.CRITICAL)

######## Reading of smiles ########


def is_smiles(text: str) -> bool:
    """
    Determine if input possibly is a smiles or just a regular text
    Tokenize the input and compare the input length vs token length
    If not the same, then a regular word or phrase.
    based on: https://github.com/pschwllr/MolecularTransformer
        Input:
            string
        Output:
            boolean, True if smiles, False if regular text
    """
    # This simple pre-check seems to work for catching plain numbers as ID (else detected as smiles)
    # normally isnumeric could be used, but if it is a number, it's from numpy and throws error.
    if not isinstance(text, str):
        return False

    pattern = "(\[[^\]]+]|Si|Ti|Al|Zn|Pd|Pt|Cu|Br?|Cl?|N|O|S|P|F|I|B|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(text)]
    length = len("".join(tokens))
    # empty could give len 0!
    # >= probably not necessary, but safer than ==
    if length and length >= len(text):
        return True
    else:
        return False


def read_smiles2pd(file_name: str, delimiter="auto", header=None) -> pd.DataFrame:
    """
    Assumes the file to be structured according to:

    SMILES in column one or two. Any additional columns are dropped.

    :param file_name: (incl path)
    :param delimiter: tab, comma, semicolon or auto: based on "priority detection". only works with , ; or tab
    :param header: None, False, True (enforces): row 0. False auto detects line 0 vs 1 if header or not
    :return: pandas table w smiles in first column & whatever identifier named as 'ID'
    """

    # determine delimiter first
    if delimiter == "auto":
        with open(file_name) as file_in:
            first_line = file_in.readline()
        if re.compile("\t").findall(first_line):
            #print("done, its a tab")
            delimiter = "\t"
        elif re.compile(";").findall(first_line):
            #print("nope, it's a ;")
            delimiter = ";"
        elif re.compile(",").findall(first_line):
            #print("ah no, its a ,")
            delimiter = ","
        # in case none is detected, potentially only one column, choose tab arbitrarily
        if delimiter == "auto":
            delimiter = "\t"

    if delimiter != "," and delimiter != ";" and delimiter != "\t":
        raise TypeError("impossible(?) delimiter given.")

    # determine if a header is present or not before finally pandas read the file
    if header is None:
        header = False  # due to later analysis

    if not header:
        # a form of header auto detection, but only for first line.
        with open(file_name) as file_in:
            first_line = file_in.readline().split(delimiter)
        first_col = first_line[0]
        # check if more than just one column
        if len(first_line) > 1:
            second_col = first_line[1]
        else:
            second_col = "dummy"
        # finally auto set header true/false
        if is_smiles(first_col) or is_smiles(second_col):
            header = False
        else:
            header = True

    if not header:
        df = pd.read_csv(
            file_name, sep=delimiter, header=None, dtype=None, engine="python"
        )
    else:
        df = pd.read_csv(
            file_name, sep=delimiter, header=0, dtype=None, engine="python"
        )
        header = True  # revert value for later analysis

    # determine smiles or text for cells 0,0 and 0,1
    _loc00 = is_smiles(df.iloc[0][0])  # smiles or text
    if len(df.columns) >= 2:
        # we don't want here other columns. becomes too much.
        df.drop(df.columns[2:], axis=1, inplace=True)
        _loc01 = is_smiles(df.iloc[0][1])  # smiles or text

    # create artificial name
    if len(df.columns) == 1:
        df[1] = "Mol" + df.index.astype(str)
        _loc01 = False
        if not _loc00:
            # loc00 is not a string, but contains the header name of the smiles
            # N.B.: we don't have headers yet; they will be moved up later, thus overwrite cell 0,1
            df.iloc[0, 1] = "ID"
        if header:
            # in case the header was enforced by user
            df.rename(columns={1: "ID"}, inplace=True)

    # for no header so far, move the first row up and set as header
    if not _loc01 and not _loc00:
        new_header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data minus the header row
        df.columns = new_header  # set the new header
        header = True  # not sure I need this anymore

    # this will switch positions of columns 0 & 1 and add correct headers
    if _loc01 and not _loc00:
        # switch the 2 columns, thus structure first, then ID
        col_list = list(df)
        if header:
            columns_titles = [col_list[1], col_list[0]]
            df = df.reindex(columns=columns_titles)
        else:
            columns_titles = [1, 0]
            df = df.reindex(columns=columns_titles)
            df.rename(columns={0: "ID", 1: "Smiles"}, inplace=True)

    # finally, the use case of auto-adding header where order is in correct place
    if not header and _loc00:
        df.rename(columns={0: "Smiles", 1: "ID"}, inplace=True)

    # coerce ID (all) to string in case it still is handled as number. to prevent later errors.
    df.iloc[0:2] = df.iloc[0:2].astype("str")

    return df


def read_rct2pd(file_name: str, delimiter="\t", header=None) -> pd.DataFrame:
    """
    Assumes the file to be structured according to:

    ID in column zero
    SMILES for A+B->C in columns one, two and three. Any additional columns are dropped.

    :param file_name: (incl path)
    :param delimiter:  optional
    :param header: not really used due to "auto" detection
    :return: pandas table w ID in first column & reaction components in 2,3 & 4
    """

    with open(file_name) as file_in:
        first_line = file_in.readline().split()
    if len(first_line) < 4:
        print("---File Error: minimum 4 columns: 'ID and A + B & Product' ---")
        return

    # column 1 as rep for compound or not
    if is_smiles(first_line[1]):
        df = pd.read_csv(file_name, sep=delimiter, header=None, dtype=None)
        df.rename(
            columns={
                0: "ID",
                1: "component1_(smiles)",
                2: "component2_(smiles)",
                3: "product_(smiles)",
            },
            inplace=True,
        )
    else:
        df = pd.read_csv(file_name, sep=delimiter, header=0, dtype=None)

    # drop columns that aren't ID or part of reaction i.e. keep only first 4 columns
    if len(df.columns) > 4:
        df.drop(df.columns[4:], axis=1, inplace=True)

    # coerce ID to string in case it still is handled as number. to prevent later errors.
    df.iloc[0:1] = df.iloc[0:1].astype("str")
    return df


def read_rct2pd_ABCD(file_name: str, delimiter="\t", header=None) -> pd.DataFrame:
    """
    Assumes the file to be structured according to:

    ID in column zero
    SMILES for A+B->C+D in columns one to four. Any additional columns are dropped.

    :param file_name: (incl path)
    :param delimiter:  optional
    :param header: not really used due to "auto" detection
    :return: pandas table w ID in first column & reaction components in 2-5
    """

    with open(file_name) as file_in:
        first_line = file_in.readline().split()
    if len(first_line) < 5:
        print("---File Error: minimum 5 columns: 'ID and A + B & D + E' ---")
        return

    # column 1 as representative if compound or not; i.e. check for header
    if is_smiles(first_line[1]):
        df = pd.read_csv(file_name, sep=delimiter, header=None, dtype=None)
        df.rename(
            columns={
                0: "ID",
                1: "component1_(smiles)",
                2: "component2_(smiles)",
                3: "product1_(smiles)",
                4: "product2_(smiles)",
            },
            inplace=True,
        )
    else:
        df = pd.read_csv(file_name, sep=delimiter, header=0, dtype=None)

    # drop columns that aren't ID or part of reaction i.e. keep only first 5 columns
    if len(df.columns) > 5:
        df.drop(df.columns[5:], axis=1, inplace=True)

    # coerce ID to string in case it still is handled as number. to prevent later errors.
    df.iloc[0:1] = df.iloc[0:1].astype("str")
    return df


""" ######## Cleaning of smiles ########
 3 (well, 4) functions in total:
 * clean_smiles - pandas of molecules stemming from input smiles. The minimum cleaner.
 * clean_all: wrapper combining clean_smiles and step2, 3, 4. Use for thoroughness or nasty datasources
 * wrapper allows for flexibility in only using one func at a time
"""


def clean_smiles(mols_raw: pd.DataFrame) -> list:
    """
    First round of cleaning of the input smiles structures.
    Creates RDKit object.
    Cleaning includes some standard normalization.
    Removes completely faulty sh** right from the start, works well for relatively good sources

    :param mols_raw: either dictionary (deprecated, though still in code), or pandas, 'preformatted' order
    :return: list containing rdkit molobjects
    """
    # NOTE: check here for details: https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization

    if not isinstance(mols_raw, dict) and not isinstance(mols_raw, pd.DataFrame):
        raise TypeError("Dictionary or pandas DataFrame expected")

    if isinstance(mols_raw, pd.DataFrame):
        mols = pd.Series(
            mols_raw.iloc[:, 0].values, index=mols_raw.iloc[:, 1]
        ).to_dict()
    else:
        mols = mols_raw

    num_entries = len(mols)
    molecule_list = []
    for name, smi in mols.items():
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)

        except ValueError as _e:
            print("ID: ", name, " ", _e)
            continue

        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES
            ),
        )
        mol = rdMolStandardize.Normalize(mol)
        # N.B. it seems that in first iteration strings as digits still are seen as
        # integers which throws an error from rdkit. this an additional conversion to str.
        # more readable, though I wonder if one could just write enforced name=str(name)?
        if not isinstance(name, str):
            name = str(name)
        mol.SetProp("_Name", name)
        molecule_list.append(mol)

    if len(molecule_list) == 0:  # "if not molecule_list:" should also work
        raise ImportError("!!! No molecules to work with!!!")
    elif len(molecule_list) < num_entries:
        print(
            "!N.B.: ",
            num_entries - len(molecule_list),
            " faulty entries have been filtered out!",
        )

    return molecule_list


def clean_step2(mol_object: list) -> list:
    """
    Neutralize, Reionize, Keep parent fragment & clean some metal bonds
    of the rdkit object

    :param mol_object: list of rdkit objects
    return: new list of rdkit objects
    """
    uncharger = rdMolStandardize.Uncharger()
    disconnector = rdMolStandardize.MetalDisconnector()
    molecule_list = []
    for mol in mol_object:
        name = mol.GetProp("_Name")
        mol = uncharger.uncharge(mol)

        try:
            uncharger.uncharge(rdMolStandardize.FragmentParent(mol))

        except ValueError as _e:
            print("ID: ", name, " ", _e)
            continue

        mol = uncharger.uncharge(rdMolStandardize.FragmentParent(mol))
        mol = disconnector.Disconnect(mol)
        # Fragmentor seems to lose the name thus need to save and set manually
        mol.SetProp("_Name", name)
        molecule_list.append(mol)

    if len(molecule_list) < len(mol_object):
        print(
            "!N.B.: ",
            len(mol_object) - len(molecule_list),
            " faulty entries have been filtered out!",
        )

    return molecule_list


def clean_step3(mol_object: list) -> list:
    """
    Neutralizes charges in a rdkit-object (containing list)
    """
    uncharger = rdMolStandardize.Uncharger()
    molecule_list = []
    for mol in mol_object:
        mol = uncharger.uncharge(mol)
        molecule_list.append(mol)

    return molecule_list


def clean_step4(mol_object: list) -> list:
    """
    Reionizes and keeps parent fragment of rdkit object(containing list)
    """
    uncharger = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    molecule_list = []
    for mol in mol_object:
        # name = mol.GetProp("_Name")
        # using name here seems to mess things up;
        # on the other hand, not necessary anyway since rdkit takes care of it at this stage
        # print("3rd round, ID: ", mol.GetProp("_Name"), " ", Chem.rdmolfiles.MolToSmiles(mol))
        mol = uncharger.uncharge(rdMolStandardize.FragmentParent(mol))
        mol = md.Disconnect(mol)

        # mol.SetProp("_Name", name)
        molecule_list.append(mol)

    return molecule_list


def clean_all(mols: pd.DataFrame) -> list:
    """
    Combines all cleaning functions into one. Could four steps cleaning be done in one step?
    Yes, but this is more flexible, can in principle skip one or more steps.
    Use for thoroughness or for nasty datasources

    :param mols: pandas df containing smiles strings
    :return: list of rdkit objects
    """

    _x0 = clean_smiles(mols)
    _x1 = clean_step2(_x0)
    _x2 = clean_step3(_x1)
    _x3 = clean_step4(_x2)

    return _x3


def smi2sdf(smiles_file: str, sdf_out: str) -> None:
    """
    Simple converter of smiles file to an sdf file.
    No error checking is done. Something to work with upon occasion
    Duplicate IDs are filtered, even if different structure.

    :param smiles_file: file name (with path) containing smiles
    :param sdf_out: file name (with path) for the sdf output
    :return: None (for now)
    """
    print("\nReading molecules from: ", smiles_file)
    in_mols = read_smiles2pd(smiles_file)
    mols = pd.Series(in_mols.iloc[:, 0].values, index=in_mols.iloc[:, 1]).to_dict()

    molecule_list = []
    for name, smi in mols.items():
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        # here could be changed for initial cleaning
        if not isinstance(name, str):
            name = str(name)
        mol.SetProp("_Name", name)
        molecule_list.append(mol)

    # write as single sdf file
    print("\nWriting molecules to: ", sdf_out)

    with Chem.SDWriter(sdf_out) as w:
        for m in molecule_list:
            w.write(m)

    return None


def default_conformation(mol_list: list, keepHs=False) -> list:
    """
    generate ETKDG default conformer, list based

    Functional and working, but
    DOESN'T WORK AS INTENDED in context of other workflows

    :param mol_list: input list of rdkit object structures
    :param keepHs: False = remove Hs (default), True, keep.
    :return: list of default conformer rdkit objects
    """
    from rdkit.Chem import AllChem

    conformers = []
    for mol in mol_list:
        # name = mol.GetProp("_Name")
        molH = Chem.AddHs(mol)
        # molH.SetProp("_Name", name)
        conformers.append(AllChem.EmbedMolecule(molH, randomSeed=42))

    out_list = []
    if keepHs:
        out_list = conformers
    else:
        for m in conformers:
            out_list.append(Chem.RemoveHs(m))

    return out_list


######### other functions and constants #############


def mordred2pandas(_) -> pd.DataFrame:
    """
    mordred itself returns a pandas compatible dataframe, though not "100%",
    thus a converter causing less issues this way

    :param _: mordred dataframe
    :return: pandas dataframe
    """
    return pd.DataFrame.from_dict(_.to_dict(), orient="columns")


class Timer:
    """
    e.g.
    t=Timer()
    codeblock
    t.log()
    """

    # time.perfcounter() better than time.time()?
    def __init__(self):
        self.start = time.time()

    def start(self):
        self.start = time.time()

    def log(self):
        logger = time.time() - self.start
        print("Time log -", logger)

    def milestone(self):
        self.start = time.time()


RDKIT_DESCRIPS = MolecularDescriptorCalculator(
    [
        "MolLogP",
        "MolMR",
        "MolWt",
        "NHOHCount",
        "NOCount",
        "FractionCSP3",
        "RingCount",
        "NumAliphaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumAliphaticRings",
        "NumAromaticCarbocycles",
        "NumAromaticHeterocycles",
        "NumAromaticRings",
        "NumHAcceptors",
        "NumHDonors",
        "NumHeteroatoms",
        "NumRadicalElectrons",
        "NumRotatableBonds",
        "NumSaturatedCarbocycles",
        "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "NumValenceElectrons",
        "HeavyAtomCount",
        #       "HeavyAtomMolWt",
        "TPSA",
        "MaxEStateIndex",
        "BalabanJ",
        "BertzCT",
        "Chi0",
        "Chi0n",
        "Chi0v",
        "Chi1",
        "Chi1n",
        "Chi1v",
        "Chi2n",
        "Chi2v",
        "Chi3n",
        "Chi3v",
        "Chi4n",
        "Chi4v",
        "EState_VSA1",
        "EState_VSA10",
        "EState_VSA11",
        "EState_VSA2",
        "EState_VSA3",
        "EState_VSA4",
        "EState_VSA5",
        "EState_VSA6",
        "EState_VSA7",
        "EState_VSA8",
        "EState_VSA9",
        #        "ExactMolWt",
        #        "FpDensityMorgan1",
        #        "FpDensityMorgan2",
        #        "FpDensityMorgan3",
        "HallKierAlpha",
        "Ipc",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        "LabuteASA",
        "MaxAbsEStateIndex",
        "MaxAbsPartialCharge",
        "MaxPartialCharge",
        "MinAbsEStateIndex",
        "MinAbsPartialCharge",
        "MinEStateIndex",
        "MinPartialCharge",
        "PEOE_VSA1",
        "PEOE_VSA10",
        "PEOE_VSA11",
        "PEOE_VSA12",
        "PEOE_VSA13",
        "PEOE_VSA14",
        "PEOE_VSA2",
        "PEOE_VSA3",
        "PEOE_VSA4",
        "PEOE_VSA5",
        "PEOE_VSA6",
        "PEOE_VSA7",
        "PEOE_VSA8",
        "PEOE_VSA9",
        "qed",
        "SlogP_VSA1",
        "SlogP_VSA10",
        "SlogP_VSA11",
        "SlogP_VSA12",
        "SlogP_VSA2",
        "SlogP_VSA3",
        "SlogP_VSA4",
        "SlogP_VSA5",
        "SlogP_VSA6",
        "SlogP_VSA7",
        "SlogP_VSA8",
        "SlogP_VSA9",
        "SMR_VSA1",
        "SMR_VSA10",
        "SMR_VSA2",
        "SMR_VSA3",
        "SMR_VSA4",
        "SMR_VSA5",
        "SMR_VSA6",
        "SMR_VSA7",
        "SMR_VSA8",
        "SMR_VSA9",
        "VSA_EState1",
        "VSA_EState10",
        "VSA_EState2",
        "VSA_EState3",
        "VSA_EState4",
        "VSA_EState5",
        "VSA_EState6",
        "VSA_EState7",
        "VSA_EState8",
        "VSA_EState9",
    ]
)

RDKIT_DESCRIPS_HEADERS = list(RDKIT_DESCRIPS.GetDescriptorNames())


"""
for MQNs there is no simple way in Python as above, thus:
use in code:

from rdkit.Chem import rdMolDescriptors
ds = rdMolDescriptors.MQNs_(m)

and attach the header list as required
"""
RDKIT_MQN_HEADERS = [
    "MQN1",
    "MQN2",
    "MQN3",
    "MQN4",
    "MQN5",
    "MQN6",
    "MQN7",
    "MQN8",
    "MQN9",
    "MQN10",
    "MQN11",
    "MQN12",
    "MQN13",
    "MQN14",
    "MQN15",
    "MQN16",
    "MQN17",
    "MQN18",
    "MQN19",
    "MQN20",
    "MQN21",
    "MQN22",
    "MQN23",
    "MQN24",
    "MQN25",
    "MQN26",
    "MQN27",
    "MQN28",
    "MQN29",
    "MQN30",
    "MQN31",
    "MQN32",
    "MQN33",
    "MQN34",
    "MQN35",
    "MQN36",
    "MQN37",
    "MQN38",
    "MQN39",
    "MQN40",
    "MQN41",
    "MQN42",
]


####################################################
#### deprecated          ####
#### keeping as reminder ####

'''
def read_smiles2dict(file_name: str, delimiter="\t", header=False) -> dict:
    """
    !!!!!DEPRECATED!!!!!

    Will only use first two columns of a file, Smiles & ID, discards all else.
    May also be order of ID & SMILES.
        Auto detection if header or not.
        pandas version might be superior?
    No real advantage of using this over the pandas right away.
        Input: filename (incl path); optional: delimiter & header. Header is 'auto'.
        Output: dictionary of smiles + name (name = key).
        Any header is discarded.
    """

    molecule_struct = []
    molecule_name = []
    counter = 0

    for line in open(file_name):
        num_columns = len(re.findall(delimiter, line))
        if num_columns >= 2:
            raise ValueError("Too many columns, use the pandas function")
            return None

        if header:
            # skip the first line
            header = False
            continue

        if num_columns == 1:
            # if there are two columns
            smi, name = line.strip().split(delimiter)
            counter += 1
            if not is_smiles(smi) and not is_smiles(name):
                # corresponds to a header
                continue
            if is_smiles(name):
                # switch for correct order
                _tmp = smi
                smi = name
                name = _tmp
            if name in molecule_name:
                name = name + "_dup"

        if num_columns == 0:
            # if only smiles, create artificial mol name/ID
            smi = line.rstrip()
            name = "Mol" + str(counter)
            counter += 1

        molecule_struct.append(smi)
        molecule_name.append(name)
    # the return dict is a dict but not in the sense of key:value
    # which renders using converters (e.g. pd to dict_list a pain)
    # an additional reason this is abandoned. leaving it here for learnings.
    return dict(zip(molecule_name, molecule_struct))
'''