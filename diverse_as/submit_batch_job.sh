for exp in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$exp" \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp = $exp; run; exit;"
done
