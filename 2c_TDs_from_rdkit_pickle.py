#!/usr/bin/env python
# coding: utf-8
"""
Add TDs to the data
"""

import argparse
import os
import pickle
import sys
from os.path import expanduser

import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Important, or else waaaay too many RDkit details in output
RDLogger.logger().setLevel(RDLogger.CRITICAL)

sys.path.insert(1, os.path.join(expanduser("~"), "dev/chemistry/"))

from mymodules.chemtools import Timer
from mymodules.rxntools import (
    clean_smiles_multi,
    elemental_tds_multi,
    rdkit_descriptors_multi,
    table_delta,
    transform_descriptors_multi,
)


def main():
    #
    # Initialization
    #
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            """\TDs
        """
        ),
    )

    # Comment next line out when editing file & not using cmd line; also uncomment the two after
    # parser.add_argument("file", type=str)  # <- comment/uncomment

    # Comment next two lines when using cmd line and uncomment above line
    data_path = os.path.join(
        "/home/alex/dev/ML-pKa/datasets_AM/"
    )  # <- uncomment/comment
    file_in = "all_data_cleaned_properties.pkl"  # <- uncomment/comment

    args = parser.parse_args()
    if not ("data_path" in locals()):
        file_in = os.path.basename(args.file)
        data_path = os.path.dirname(args.file)
        if data_path == "" or data_path == ".":
            data_path = os.getcwd()

    file_base = os.path.splitext(file_in)[0]
    # overwrite file_base
    # file_base = "avlilumove_cleaned"
    # file_base = "novartis_cleaned"
    sdf_out = os.path.join(data_path + file_base + "_TDs.sdf")
    pickle_out = os.path.join(data_path + file_base + "_TDs.pkl")
    tsv_out = os.path.join(data_path + file_base + "_TDs.tsv")

    pickle_input = os.path.join(data_path + file_in)

    # read pickle file
    print("\nReading pickle file: ", pickle_input)
    with open(pickle_input, "rb") as f:
        in_mols = pickle.load(f)

    if not ("ROMol" in in_mols.columns):
        in_mols["ROMol"] = in_mols["ISO_SMI"].apply(Chem.MolFromSmiles)

    # clean_smiles = clean_smiles_multi(listofsmiles)
    clean_smiles = in_mols["ISO_SMI"].tolist()
    eas0 = elemental_tds_multi(clean_smiles)
    # rdk = rdkit_descriptors_multi(clean_smiles)
    tds1 = transform_descriptors_multi(clean_smiles)
    final_numbers = pd.concat([in_mols, eas0, tds1], axis=1, join="inner")
    merged_df = final_numbers

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


if __name__ == "__main__":
    main()
