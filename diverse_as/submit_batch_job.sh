export data=ecfp1
export group_size=4

for exp in {1..7}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$exp" -m "mangosteen.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data'; run; exit;"
done
