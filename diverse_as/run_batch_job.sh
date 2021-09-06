export policy="ens jensen greedy"

for exp in {1..2}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=10]" \
  -o job_output/run.%J -J "run$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp = $exp; run_square; exit;"
done
