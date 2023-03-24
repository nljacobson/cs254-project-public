import numpy as np 
import copy

from sklearn.linear_model import LogisticRegression

import features as f 
import pandas as pd 
import sklearn as sk
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from logistic_regression import hc_logistic_regression


class Solution:
    def __init__(self, cluster_size, fitness, min_samples_leaf, min_samples_split, max_features):
        self.genome = []
        self.fitness = fitness 
        self.cluster_size = cluster_size
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def set_genome(self, genome):
        self.genome = genome

    def is_already_in_genome(self, new_feature):  
        flattened_genome = [item for sublist in self.genome for item in sublist]
        if new_feature in flattened_genome:
            return True 
        return False
    
    def mutate(self):
        new_hyperparams = [self.min_samples_leaf + np.random.randint(-2,2), self.min_samples_split+ np.random.randint(-2,2),self.max_features+ np.random.randint(-2,2)]
        if new_hyperparams[2] > self.cluster_size:
            new_hyperparams[2] = self.cluster_size -1
        for i in range(len(new_hyperparams)):
            if new_hyperparams[i] <= 1:
                new_hyperparams[i] = 2
            if new_hyperparams[i] > 100:
                new_hyperparams[i] = 100
        new_gen = self.genome.copy()
        not_finished = True 
        
        while not_finished:
            new_feature = np.random.choice(f.wave_one_features[1:] + f.wave_two_features[1:])
            # if self.is_already_in_genome(new_gen) == False:
            new_gen[np.random.randint(self.cluster_size)] = new_feature
            not_finished = False
        return new_gen, new_hyperparams


class ParallelHillClimber:
    def __init__(self, pop_size, num_gens, cluster_size, fitness_file, seed, ml_model):
        self.seed = seed 
        self.ml_model = ml_model
        np.random.seed(seed)
        # os.system("del fitness.csv") 
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

        data_all1.loc[data_all1['H4TO5'] >=25, 'H4TO5'] = 30
        data_all1.loc[data_all1['H4TO5'] >=31, 'H4TO5'] = 'NaN'
        data_all1.loc[data_all1['H4TO5'].between(5,25), 'H4TO5'] = 15
        data_all1.loc[data_all1['H4TO5'] <5, 'H4TO5'] = 0
        data_all1.dropna(subset=['H4TO5'])
        for col in data_all1:
            data_all1[col].fillna(data_all1[col].mode()[0], inplace=True)
        return data_all1
    
    def evolve(self):
        f = open(self.fitness_file, "w")
        f.write("{},{},{},{},{},{}\n".format("curr_gen", "fitness", "genome","min_samples_leaf", "min_samples_split", "max_features"))
        f.close()
        self.population = self.create_initial_population()
        for i in range(self.num_gens):
            self.evolve_one_generation()
            self.record_best(i)
        return self.best.fitness 

    def record_best(self, curr_gen):
        '''records best genome in the fitness file'''
        self.best = sorted(self.population, key=lambda x: x.fitness, reverse=True)[0]
        f = open(self.fitness_file, "a")
        f.write("{},{},{},{},{},{}\n".format(curr_gen, self.best.fitness, self.best.genome, self.best.min_samples_leaf, self.best.min_samples_split, self.best.max_features))
        f.close()

    def evolve_one_generation(self):   
        '''does one generation of evolution'''
        for solution in self.population:    
            new_genome, new_hyperparams = self.mutate(solution)
            new_fitness = self.get_fitness(new_genome, new_hyperparams)
            if new_fitness > solution.fitness:
                solution.set_genome(new_genome)
                solution.min_samples_leaf=new_hyperparams[0]
                solution.min_samples_split=new_hyperparams[1]
                solution.max_features=new_hyperparams[2]
                solution.fitness = new_fitness

    def create_initial_population(self):
        '''Creates a population of solutions of size pop_size
        returns: population created'''

        population = []
        for i in range(self.pop_size):
            min_samples_leaf=np.random.randint(2,100)
            min_samples_split=np.random.randint(3,100)
            max_features=np.random.randint(2,self.cluster_size)
            genome = list(np.random.choice(self.all_features, size = self.cluster_size, replace = False))
            fitness = self.get_fitness(genome, [min_samples_leaf,min_samples_split, max_features])
            sln = Solution(self.cluster_size, fitness =fitness,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split, max_features=max_features )
            sln.set_genome(genome)
            population.append(sln)
        return population

    def mutate(self, solution):
        '''mutates one the genome of a group of features
        returns: mutated genome'''
        new_genome, new_hyperparams = solution.mutate()
        return new_genome, new_hyperparams

    def get_fitness(self, genome, hyperparams):
        '''get the preformance of a given genome 
        for the ML model of interest (currently decision tree classifier
        returns: preformance score'''
        
        x_train, x_test, y_train, y_test = train_test_split(self.data[genome], self.data.iloc[:, 226:], test_size = 0.33, random_state = self.seed)         
        if self.ml_model == "random_forest":                         
            rf= RandomForestClassifier(random_state= self.seed,min_samples_leaf=hyperparams[0], min_samples_split=hyperparams[1], max_features= hyperparams[2]) 
            rf.fit(x_train, np.ravel(y_train))
            score= rf.score(x_test, y_test)
        if self.ml_model == "decision_tree":    
            decision_tree= DecisionTreeClassifier(random_state= self.seed,min_samples_leaf=hyperparams[0], min_samples_split=hyperparams[1], max_features= hyperparams[2]) 
            decision_tree.fit(x_train, np.ravel(y_train))
            score= decision_tree.score(x_test, y_test)
        if self.ml_model == "log_regression":
            #xtrain = x_train.to_numpy(dtype='int')
            #ytrain = y_train.to_numpy(dtype='int').reshape(len(y_train), 1)
            #xtest = x_test.to_numpy()
            #ytest= y_test.to_numpy().flatten()
            #score = hc_logistic_regression(xtrain, ytrain, xtest, ytest, eta=.1, iters = 10, threshold=25)
            clf = LogisticRegression(random_state=0, max_iter=1000)
            clf.fit(x_train, y_train.astype('int32'))
            score = clf.score(x_test, y_test.astype('int32').to_numpy().reshape(len(y_test)))
        return score


phc_l = ParallelHillClimber(ml_model="log_regression", pop_size=100, num_gens=300, cluster_size=10, fitness_file="fitness.csv", seed=0)
best_l = phc_l.evolve()
print(best_l)

