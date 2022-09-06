for exp in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  -o bjob_output/vast.%J -J "c-$group_size-$exp" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; data='vast'; run; exit;"
done
