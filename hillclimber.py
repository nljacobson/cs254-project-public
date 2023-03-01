import numpy as np 
import copy 
import features as f 
import pandas as pd 
import sklearn as sk
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

class Solution:
    def __init__(self, cluster_size, fitness):
        self.genome = []
        self.fitness = fitness 
        self.cluster_size = cluster_size

    def set_genome(self, genome):
        self.genome = genome

    def is_already_in_genome(self, new_feature):  
        flattened_genome = [item for sublist in self.genome for item in sublist]
        if new_feature in flattened_genome:
            return True 
        return False
    
    def mutate(self):
        new_gen = self.genome.copy()
        not_finished = True 
        
        while not_finished:
            new_feature = np.random.choice(f.wave_one_features[1:] + f.wave_two_features[1:])
            if self.is_already_in_genome(new_gen) == False:
                new_gen[np.random.randint(self.cluster_size)] = new_feature
                not_finished = False
        return new_gen


class ParallelHillClimber:
    def __init__(self, pop_size, num_gens, cluster_size, fitness_file):
        os.system("del fitness.csv")
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.cluster_size = cluster_size
        self.fitness_file = fitness_file
        self.all_features = f.wave_one_features[1:] + f.wave_two_features[1:]
        self.data = self.gen_df()

    def gen_df(self):
        '''creates data base'''
        df1 = pd.read_csv('data/wave_1/21600-0001-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_one_features)
        df2 = pd.read_csv('data/wave_2/21600-0005-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_two_features)
        df4 = pd.read_csv('data/wave_4/21600-0022-Data.tsv', sep='\t', header=0, low_memory=False, usecols=f.wave_four_outcomes)

        data_all_features= pd.merge(df1, df2, on="AID", how= "outer")
        data_all= pd.merge(data_all_features, df4, on="AID", how= "right")

        data_all1= data_all.replace(r'^\s*$',np.nan, regex=True)
        data_all1= data_all1.astype(float)

        for col in data_all1:
            data_all1[col].fillna(data_all1[col].mode()[0], inplace=True)
        return data_all1
    
    def evolve(self):
        self.population = self.create_initial_population()
        for i in range(self.num_gens):
            self.evolve_one_generation()
            self.record_best(i)
        return self.best.fitness 

    def record_best(self, curr_gen):
        '''records best genome in the fitness file'''
        self.best = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]
        f = open(self.fitness_file, "a")
        f.write("{},{},{}\n".format(curr_gen, self.best.fitness, self.best.genome))
        f.close()

    def evolve_one_generation(self):   
        '''does one generation of evolution'''
        for solution in self.population:    
            new_genome = self.mutate(solution)
            new_fitness = self.get_fitness(new_genome)
            if new_fitness > solution.fitness:
                solution.set_genome(new_genome)
                solution.fitness = new_fitness

    def create_initial_population(self):
        '''Creates a population of solutions of size pop_size
        returns: population created'''

        population = []
        for i in range(self.pop_size):
            genome = list(np.random.choice(self.all_features, size = self.cluster_size, replace = False))
            fitness = self.get_fitness(genome)
            sln = Solution(self.cluster_size, fitness)
            sln.set_genome(genome)
            population.append(sln)
        return population

    def mutate(self, solution):
        '''mutates one the genome of a group of features
        returns: mutated genome'''
        new_genome = solution.mutate()
        return new_genome

    def get_fitness(self, genome):
        '''get the preformance of a given genome 
        for the ML model of interest (currently decision tree classifier
        returns: preformance score'''
        x_train, x_test, y_train, y_test = train_test_split(self.data[genome], self.data.iloc[:, 226:], test_size = 0.33, random_state = 0)
        decision_tree= DecisionTreeClassifier(random_state= 0)
        decision_tree.fit(x_train, y_train)
        score= decision_tree.score(x_test, y_test)
        return score 

phc = ParallelHillClimber(pop_size=2, num_gens=2, cluster_size=10,fitness_file="fitness.csv")
best = phc.evolve()