from hillclimber import ParallelHillClimber
import sys

seed = int(sys.argv[1])
cluster_size = int(sys.argv[2])

phc = ParallelHillClimber(pop_size=100, num_gens=300, cluster_size=cluster_size,fitness_file="hc_data/fitness_cluster_size{}_seed{}.csv".format(cluster_size, seed), seed = seed)
best = phc.evolve()