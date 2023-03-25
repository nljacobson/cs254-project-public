from hillclimber import ParallelHillClimber
import sys

seed = int(sys.argv[1])
cluster_size = int(sys.argv[2])
ml_model = sys.argv[3]
phc = ParallelHillClimber(pop_size=10, num_gens=300, cluster_size=cluster_size,fitness_file="hc_data/{}_cluster_size{}_seed{}.csv".format(ml_model, cluster_size, seed), seed = seed, ml_model =ml_model)
best = phc.evolve()