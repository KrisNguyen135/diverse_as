export group_size=1

for exp in {1..1}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/bmg.%J -J "b-$group_size-$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='bmg'; run; exit;"
done
