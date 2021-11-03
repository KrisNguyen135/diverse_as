for exp in {1..2}
do
  bsub -q "centos8" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run_citeseer.%J -J "citeseer$exp" \
  -m "node13.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "policy='classical ens'; exp=$exp; data='citeseer'; run; exit;"
done
