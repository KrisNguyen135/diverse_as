# export data=ecfp2
# export group_size=4
#
# for exp in {1..5}
# do
#   bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
#   -o bjob_output/run.%J -J "run$exp" -m "kumquat.engr.wustl.edu" \
#   matlab -nodesktop -nosplash -nodisplay -r \
#   "exp=$exp; group_size=$group_size; data='$data'; run; exit;"
# done

# -R "hname!=rambutan.engr.wustl.edu" \
# -m "mangosteen.engr.wustl.edu" \

export data=morgan
export group_size=9
export exp=1

for group in {1..20}
do
  bsub -q "normal" -R "rusage[mem=20]" -o bjob_output/drug.%J -J "$group_size-$group" -g /quan/cpu \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data$group'; run; exit;"
done
