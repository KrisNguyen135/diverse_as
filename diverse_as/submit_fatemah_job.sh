for exp in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/fatemah.%J -J "c-$group_size-$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; run_fatemah; exit;"
done
