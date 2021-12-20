#BSUB -q centos8
#BSUB -G SEAS-Lab-Garnett
#BSUB -o bjob_output/nn.%J
#BSUB -J nn
#BSUB -R "rusage[mem=120G]"

sleep 5
ls /storage1/garnett
sleep 5

matlab -nodesktop -nosplash -nodisplay -r "group_size = 4; calculate_nearest_neighbors_all_classes; exit;"
