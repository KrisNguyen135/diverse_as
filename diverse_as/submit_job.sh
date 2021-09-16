export exp=1
export node=kumquat

bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
-o bjob_output/run.%J -J "run$exp" -m "$node.engr.wustl.edu" \
matlab -nodesktop -nosplash -nodisplay -r \
"exp = $exp; run; exit;"
