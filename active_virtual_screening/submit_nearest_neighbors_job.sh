#BSUB -q centos8
#BSUB -G SEAS-Lab-Garnett
#BSUB -o bjob_output/nn.%J
#BSUB -J nn
#BSUB -m "node01.engr.wustl.edu"
#BSUB -R "rusage[mem=120G]"

sleep 5
ls /storage1/garnett
sleep 5

matlab -nodesktop -nosplash -nodisplay -r "calculate_nearest_neighbors_all_classes; exit;"
