import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default="demo_real_iter1")
parser.add_argument("--policy", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=100000)
parser.add_argument("--score_threshold", type=float, default=None)
args = parser.parse_args()

data_name = args.data_name
demo, iteration = data_name.split("real_iter")
iteration = int(iteration)
policy = args.policy
sample_size = args.sample_size
iter_dir = "iteration%d" % iteration
iter_dir = demo + iter_dir

smiles_file = '%s/data/smiles_sample_size_%d_iter%d' % (iter_dir, sample_size, iteration)
label_file = '%s/data/labels' % (iter_dir)
labels = np.loadtxt(label_file, dtype=int)
num_labeled = len(labels)
num_positve = np.sum(labels)

rec_dir = "%s/recommended_batch/" % iter_dir
chosen_ind_file = rec_dir + '%s_policy_%d_chosen_ind' % (data_name, policy)
recommended_smiles_file = rec_dir + 'recommended_smiles_iter%d' % iteration
recommended_ind_file = rec_dir + 'recommended_ind_iter%d' % iteration

print("reading smiles file: %s" % smiles_file)
with open(smiles_file, 'r') as f:
    lines = f.readlines()
print("sample size: %d" % len(lines))

chosen_ind = np.loadtxt(chosen_ind_file, dtype=int)


if args.score_threshold:
    greedy_chosen_scores_file = rec_dir + '%s_scores' % data_name
    greedy_chosen_scores = np.loadtxt(greedy_chosen_scores_file, dtype=float)
    meaningful_ind = greedy_chosen_scores > num_positive + args.score_threshold  # 0.004 is the prior probability
    chosen_ind = chosen_ind[meaningful_ind]
print("#points chosen: %d" % len(chosen_ind))

python_chosen_ind = chosen_ind - num_labeled - 1  # python index starts from zero
np.savetxt(recommended_ind_file, python_chosen_ind, fmt="%d")
print(python_chosen_ind)
with open(recommended_smiles_file, 'w') as f:
    for i, ind in enumerate(python_chosen_ind):
        f.write(lines[ind])
print("recommended smiles file path: %s" % recommended_smiles_file)