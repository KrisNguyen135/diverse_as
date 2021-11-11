# export data=ecfp2
# export group_size=4
#
# for exp in {1..5}
# do
#   bsub -q "normal" -G SEAS-Lab-Garnett -R "rusage[mem=20]" \
#   -o bjob_output/run.%J -J "run$exp" -m "kumquat.engr.wustl.edu" \
#   matlab -nodesktop -nosplash -nodisplay -r \
#   "exp=$exp; group_size=$group_size; data='$data'; run; exit;"
# done

# -R "hname!=rambutan.engr.wustl.edu && hname!=soursop.engr.wustl.edu && hname!=kumquat.engr.wustl.edu && hname!=mangosteen.engr.wustl.edu && hname!=node01.engr.wustl.edu" \
# -m "mangosteen.engr.wustl.edu"

export data=morgan
export group_size=1
export exp=1

for group in {38..38}
do
  bsub -q "normal" -R "rusage[mem=20]" \
  -o bjob_output/run.%J -J "run$group" \
  -R "hname!=rambutan.engr.wustl.edu" \
  -R "hname!=soursop.engr.wustl.edu" \
  -R "hname!=mangosteen.engr.wustl.edu" \
  -R "hname!=node02.engr.wustl.edu" \
  -R "hname!=gnode01.engr.wustl.edu" \
  -R "hname!=gnode02.engr.wustl.edu" \
  -R "hname!=gnode05.engr.wustl.edu" \
  -R "hname!=lotus.engr.wustl.edu" \
  -R "hname!=node01.engr.wustl.edu" \
  -R "hname!=node13.engr.wustl.edu" \
  -R "hname!=gnode01.engr.wustl.edu" \
  -R "hname!=gnode02.engr.wustl.edu" \
  -R "hname!=gnode05.engr.wustl.edu" \
  -R "hname!=lotus.engr.wustl.edu" \
  -R "hname!=node16.engr.wustl.edu" \
  -R "hname!=node28.engr.wustl.edu" \
  -R "hname!=node24.engr.wustl.edu" \
  -R "hname!=node08.engr.wustl.edu" \
  -R "hname!=node07.engr.wustl.edu" \
  matlab -nodesktop -nosplash -nodisplay -r \
  "exp=$exp; group_size=$group_size; data='$data$group'; run; exit;"
done
