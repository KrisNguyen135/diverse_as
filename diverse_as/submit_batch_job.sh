export node=mangosteen

for exp in {1..5}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$exp" -m "$node.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp = $exp; run; exit;"
done
