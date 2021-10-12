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

export data=ecfp
export group_size=4

for group in {4..10}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$group" -m "kumquat.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "group_size=$group_size; data='$data$group'; run; exit;"
done
