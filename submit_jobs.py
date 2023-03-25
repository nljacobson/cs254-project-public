import os 

seed = 0
# ml_model = "decision_tree"
# ml_model = "random_forest"
ml_model = "log_regression"
for cluster_size in range(10, 100, 5):
    os.system("python begin_hc.py {} {} {}".format(seed, cluster_size, ml_model))
    print(cluster_size)