import os 

seed = 0
ml_model = "decision_tree"
for cluster_size in range(10, 100, 5):
    os.system("sbatch submit_hc.sh {} {} {}".format(seed, cluster_size, ml_model))