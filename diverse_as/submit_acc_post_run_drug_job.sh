export data=morgan
export group_size=14
export exp=1

for group in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -o bjob_output/acc_drug.%J -J "acc-d-$group_size-$group" -g /quan/cpu \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data$group'; acc_post_run; exit;"
done
