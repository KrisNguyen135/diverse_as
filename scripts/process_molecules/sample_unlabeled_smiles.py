from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import time
import os
from utils import compute_morgan_features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--demo", type=bool, default=False)
parser.add_argument("--iteration", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=100000)
parser.add_argument("--print_per", type=int, default=10)
parser.add_argument(
    "--unlabeled_smiles_path",
    type=str,
    default="/media/data_2t/datasets/centaur_science/all_uniq_mols_fixed.txt",
)
parser.add_argument("--morgan_radius", type=int, default=2)
parser.add_argument("--morgan_nbits", type=int, default=2048)
args = parser.parse_args()

iteration = args.iteration  # change this accordingly
sample_size = args.sample_size  # sample 100k
print_per = args.print_per
unlabeled_smiles_path = args.unlabeled_smiles_path

save_dir = f"iteration{iteration}/data"

if args.demo:
    save_dir = f"demo_iteration{iteration}/data"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

feature_filepath = f"{save_dir}/features_unlabeled"
# to save the sampled smiles
smiles_filepath = f"{save_dir}/smiles_sample_size_{sample_size}_iter{iteration}"
idx_filepath = f"{save_dir}/sample_idx_iter{iteration}.pkl"

# read unlabeled data
print("reading smiles...")
tic = time.time()
with open(unlabeled_smiles_path, "r") as f:
    unlabeled_smiles = f.readlines()
toc = time.time()
print("reading complete, took %.2f seconds" % (toc - tic))
num_total_unlabeled = len(unlabeled_smiles)

# sample 100k
np.random.seed(0)  # don't change the seed there
permutation = np.random.permutation(num_total_unlabeled)
# sample a different set of unlabeled smiles in each iteration
sample_idx = permutation[sample_size * (iteration - 1) : sample_size * iteration]
with open(idx_filepath, "bw") as f:
    pickle.dump({"sample_idx": sample_idx}, f)
print("sampled indices file path: %s" % idx_filepath)

smiles_list = [unlabeled_smiles[i].strip() for i in list(sample_idx)]
print("computing morgan features and saving to path: %s" % feature_filepath)
compute_morgan_features(
    smiles_list,
    feature_filepath,
    smiles_filepath,
    radius=args.morgan_radius,
    nbits=args.morgan_nbits,
)
print("sampled smiles printed to: %s" % smiles_filepath)
