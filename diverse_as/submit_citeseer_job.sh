export group_size=9

for exp in {1..20}
do
  bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" -g /quan/cpu \
  -o bjob_output/citeseer.%J -J "c-$group_size-$exp" \
  -m "mangosteen.engr.wustl.edu kumquat.engr.wustl.edu soursop.engr.wustl.edu rambutan.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='citeseer'; run; exit;"
done
