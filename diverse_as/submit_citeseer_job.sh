for exp in {1..1}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run_citeseer.%J -J "citeseer$exp" \
  -m "node03.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; data='citeseer'; run; exit;"
done
