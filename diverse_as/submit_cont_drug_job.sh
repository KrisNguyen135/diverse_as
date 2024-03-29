export data=morgan
export group_size=14
export exp=1

for group in 6
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -o bjob_output/cont_drug.%J -J "d-$group_size-$group" -g /quan/cpu \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data$group'; continue_run; exit;"
done
