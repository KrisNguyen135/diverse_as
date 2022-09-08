export group_size=9

for exp in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/acc_citeseer.%J -J "acc-c-$group_size-$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='citeseer'; acc_post_run; exit;"
done
