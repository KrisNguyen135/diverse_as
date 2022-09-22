for exp in {1..10}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/fashion.%J -J "f-$group_size-$exp" \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; data='fashion'; run; exit;"
done
