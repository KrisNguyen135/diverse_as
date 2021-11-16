export group_size=1

for exp in {21..25}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/citeseer.%J -J "citeseer$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='citeseer'; run; exit;"
done
