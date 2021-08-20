# given a labeled file where each line is of format:
# "smiles, True/False"
# compute the morgan fingerprint of the smiles string
# and write a 0/1 "labels" file

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sys
from utils import compute_morgan_features
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file_path", type=str, default="demo-active-search-list.txt"
)
parser.add_argument(
    "--feature_file_path", type=str, default="data/features"
)
parser.add_argument(
    "--labels_file_path", type=str, default="data/labels"
)
parser.add_argument("--morgan_radius", type=int, default=2)
parser.add_argument("--morgan_nbits", type=int, default=2048)
args = parser.parse_args()

print("reading smiles from: %s" % args.input_file_path)
with open(args.input_file_path, "r") as f:
    lines = f.readlines()
smiles = [line.split(", ")[0].strip() for line in lines]
labels = np.array([line.split(", ")[1].strip() == "True" for line in lines])

dirname = os.path.dirname(args.feature_file_path)
os.makedirs(dirname, exist_ok=True)
print("computing morgan feature and saving to: %s" % args.feature_file_path)
invalid_idx = compute_morgan_features(
    smiles,
    args.feature_file_path,
    radius=args.morgan_radius,
    nbits=args.morgan_nbits,
    print_per=round(len(labels) / 10),
)

labels = labels[[i for i in range(len(labels)) if i not in invalid_idx]]
dirname = os.path.dirname(args.labels_file_path)
os.makedirs(dirname, exist_ok=True)
print("saving labels file to: %s" % args.labels_file_path)
np.savetxt(args.labels_file_path, labels, fmt="%d", delimiter="\n")
