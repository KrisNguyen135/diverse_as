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
export exp=2

for group in {6..10}
do
  bsub -q "normal" -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$group" \
  -R "hname!=rambutan.engr.wustl.edu && hname!=soursop.engr.wustl.edu && hname!=kumquat.engr.wustl.edu && hname!=mangosteen.engr.wustl.edu && hname!=node01.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data$group'; run; exit;"
done
