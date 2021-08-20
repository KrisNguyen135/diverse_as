demo=1
iteration=1
sample_size=100
k=20
policy=1
batch_size=50
num_queries=2
labeled_file_path=demo-active-search-list.txt
echo "processing labeled data ...\n"
python process_labeled_data.py --input_file_path $labeled_file_path
mkdir -p demo_iteration${iteration}/data
cp demo_initial_labeled_data/labels demo_iteration${iteration}/data

echo "sample unlabeled data and compute morgan features ...\n"
python sample_unlabeled_smiles.py --demo True --iteration $iteration --unlabeled_smiles_path demo_unlabeled_smiles --sample_size $sample_size

echo "compute nearest neighbors data structure ...\n"
matlab -nodisplay -nojvm -singleCompThread -r "calculate_nearest_neighbors_include_unlabeled(1, $iteration, $k); exit"

echo "iteration $iteration/$num_queries, choosing a batch of size $batch_size with active search policy $policy\n"
data_name=real_iter$iteration
if [ $demo -eq 1 ]
then
  data_name=demo_$data_name
fi
matlab -nodisplay -nojvm -singleCompThread -r "choose_one_batch('$data_name', $policy, $batch_size, $num_queries, $k); exit"

# policy=2
# echo "iteration $iteration/$num_queries, choosing a batch of size $batch_size with active search policy $policy\n"
# matlab -nodisplay -nojvm -singleCompThread -r "choose_one_batch('$data_name', $policy, $batch_size, $num_queries, $k); exit"

echo
echo "extract smiles based on recommended indices\n"
num_labeled=$(wc -l $labeled_file_path)
python get_the_batch_of_smiles.py --data_name $data_name --policy $policy --sample_size $sample_size
