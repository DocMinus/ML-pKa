#!/usr/bin/env python
# coding: utf-8
"""
Version 1.0.0 (Created 12 Dec 2023)
Update: 2023-12-12 repurposing an older code
@author: Alexander Minidis (DocMinus)
Purpose: Adding rdkit specifically to pka datasets
Copyright (c) 2023 DocMinus

"""

import argparse
import os
import pickle
import sys
from os.path import expanduser

import pandas as pd

# Important, or else waaaay too many RDkit details in output
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from mymodules.chemtools import (
    RDKIT_DESCRIPS,
    RDKIT_DESCRIPS_HEADERS,
    RDKIT_MQN_HEADERS,
)

sys.path.insert(1, os.path.join(expanduser("~"), "dev/chemistry/"))


def rdkit_calc(mol_df: pd.DataFrame) -> pd.DataFrame:
    """
    calculates RDkit descriptors, incl. MQN, excl. some superflous ones

    :param mol_df: list of RDkit molecule objects
    :return: pandas df containing calculated properties (no structures though)
    """
    print(mol_df.head())
    mol_list = mol_df["ROMol"].tolist()  # Use column "ROMol" for mol_list
    print("\nCalculating RDkit descriptors part 1/2")
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="rdkit.Chem.Graphs")
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="rdkit.Chem.GraphDescriptors"
    )
    calcrdk1 = [RDKIT_DESCRIPS.CalcDescriptors(mol) for mol in tqdm(mol_list)]
    transposed_properties = zip(*calcrdk1)
    headers = RDKIT_DESCRIPS_HEADERS
    named_properties = dict(zip(headers, transposed_properties))
    properties_df1 = pd.DataFrame(data=named_properties)

    print("\nCalculating RDkit descriptors part 2/2")  # MQN unfortunately separate
    from rdkit.Chem import rdMolDescriptors

    calcrdk2 = [rdMolDescriptors.MQNs_(mol) for mol in tqdm(mol_list)]
    transposed_properties = zip(*calcrdk2)
    headers = RDKIT_MQN_HEADERS
    named_properties = dict(zip(headers, transposed_properties))
    properties_df2 = pd.DataFrame(data=named_properties)

    properties_df = pd.concat([properties_df1, properties_df2], axis=1)
    return properties_df


def main():
    #
    # Initialization
    #
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            """\rdkit descriptors
        """
        ),
    )

    # Comment next line out when editing file & not using cmd line; also uncomment the two after
    # parser.add_argument("file", type=str)  # <- comment/uncomment

    # Comment next two lines when using cmd line and uncomment above line
    data_path = os.path.join(
        "/home/alex/dev/ML-pKa/datasets_AM/"
    )  # <- uncomment/comment
    file_in = "all_data_cleaned.pkl"  # <- uncomment/comment

    args = parser.parse_args()
    if not ("data_path" in locals()):
        file_in = os.path.basename(args.file)
        data_path = os.path.dirname(args.file)
        if data_path == "" or data_path == ".":
            data_path = os.getcwd()

    file_base = os.path.splitext(file_in)[0]
    sdf_out = os.path.join(data_path + file_base + "_properties.sdf")
    pickle_out = os.path.join(data_path + file_base + "_properties.pkl")
    tsv_out = os.path.join(data_path + file_base + "_properties.tsv")

    pickle_input = os.path.join(data_path + file_in)

    # read pickle file
    print("\nReading pickle file: ", pickle_input)
    with open(pickle_input, "rb") as f:
        in_mols = pickle.load(f)

    # Add numerical ID column
    in_mols["ID"] = range(1, len(in_mols) + 1)

    _df = in_mols[["ID", "ROMol"]].copy()
    print(_df.head())
    #############################################################################
    properties_df = rdkit_calc(_df)
    #############################################################################
    print("\nAttaching data...")
    # Merge properties_df and in_mols based on index
    merged_df = in_mols.join(properties_df)
    print("\nMerged DataFrame:")
    print(merged_df.head())
    #############################################################################

    PandasTools.WriteSDF(
        merged_df,
        sdf_out,
        molColName="ROMol",
        properties=list(merged_df.columns),
    )

    # drop the ROMol column
    merged_df.drop("ROMol", axis=1, inplace=True)

    with open(pickle_out, "wb") as file:
        pickle.dump(merged_df, file)

    merged_df.to_csv(
        tsv_out, sep="\t", index=False, header=True, quoting=None, decimal="."
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
