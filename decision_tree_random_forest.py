import numpy as np 
import copy 
import features as f 
import pandas as pd 
import sklearn as sk
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, RandomizedSearchCV


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

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


def create_tree(crit, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
                max_features, rand, max_leaf_nodes, ccp_alpha):
    tree= DecisionTreeClassifier(criterion= crit, max_depth= max_depth, min_samples_split= min_samples_split, 
                                 min_samples_leaf= min_samples_leaf, min_weight_fraction_leaf= min_weight_fraction_leaf, 
                                 max_features= max_features, random_state= rand, max_leaf_nodes= max_leaf_nodes, ccp_alpha= ccp_alpha)
    return tree


def create_forest(n_est, crit, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
                max_features, rand, max_leaf_nodes, ccp_alpha):
    forest= RandomForestClassifier(n_estimators=n_est, criterion=crit, max_depth= max_depth, min_samples_split= min_samples_split, 
                                   min_samples_leaf= min_samples_leaf, min_weight_fraction_leaf= min_weight_fraction_leaf, 
                                   max_features= max_features, random_state= rand, max_leaf_nodes= max_leaf_nodes, ccp_alpha= ccp_alpha)
    return forest


def calc_rocauc_recall(classifier, x_test, y_test):
    y_pred= classifier.predict(x_test)
    y_score= classifier.predict_proba(x_test)
    roc_auc= roc_auc_score(y_test, y_score, multi_class='ovr')
    recall= recall_score(y_test, y_pred, average=None)
    return roc_auc, recall


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
        
        #converting output column to binary
        #30= daily smoker (25+ out of last 30)
        #15= casual smoker (5-24 out of last 30)
        #0= nonsmoker (less than 5 days ouf ot last 30)
        data_all1.loc[data_all1['H4TO5'] >=25, 'H4TO5'] = 30
        data_all1.loc[data_all1['H4TO5'].between(5,25), 'H4TO5'] = 15
        data_all1.loc[data_all1['H4TO5'] <5, 'H4TO5'] = 0

        #missing vals filled with mode
        for col in data_all1:
            data_all1[col].fillna(data_all1[col].mode()[0], inplace=True)
        return data_all1
    
    
    def evolve(self):
        self.population = self.create_initial_population()
        for i in range(self.num_gens):
            self.evolve_one_generation()
            self.record_best(i)
        return self.best.fitness 
    
    def evolve_dtree_params(self):
        self.population = self.create_initial_population()
        for i in range(self.num_gens):
            self.evolve_one_generation_dtree_params()
            self.record_best(i)
        roc_auc, recall= calc_rocauc_recall(self.best.tree, self.best.x_test, self.best.y_test)
        return self.best.fitness, self.best.tree, roc_auc, recall, self.best.tree.get_params(), self.best.x_test, self.best.y_test, self.best.x_train, self.best.y_train
    
    def evolve_rforest_params(self):
        self.population = self.create_initial_population()
        for i in range(self.num_gens):
            self.evolve_one_generation_rforest_params()
            self.record_best(i)
        roc_auc, recall= calc_rocauc_recall(self.best.forest, self.best.x_test, self.best.y_test)
        return self.best.fitness, self.best.forest, roc_auc, recall, self.best.params, self.best.x_test, self.best.y_test, self.best.y_train
    
    '''
    def evolve_dtree_params_make_graph(self):
        self.population = self.create_initial_population()
        X= []
        Y= []
        for i in range(self.num_gens):
            self.evolve_one_generation_dtree_params()
            self.record_best(i)
            X.append(i)
            Y.append(self.best.fitness)
        plot= plt.plot(X,Y)
        plt.xlabel('Generation')
        plt.ylabel('Score')
        return plot, self.best.fitness
    '''

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
                
                
    def evolve_one_generation_dtree_params(self):   
        '''does one generation of evolution'''
        for solution in self.population:    
            new_genome = self.mutate(solution)
            new_tree = self.get_fitness_tree_params(new_genome)
            new_fitness = new_tree[0]
            if new_fitness > solution.fitness:
                solution.set_genome(new_genome)
                solution.tree = new_tree[1]
                solution.x_test= new_tree[2]
                solution.y_test= new_tree[3]
                solution.x_train= new_tree[4]
                solution.y_train= new_tree[5]
                
    def evolve_one_generation_rforest_params(self):   
        '''does one generation of evolution'''
        for solution in self.population:    
            new_genome = self.mutate(solution)
            new_forest = self.get_fitness_forest_params(new_genome)
            new_fitness = new_forest[0]
            if new_fitness > solution.fitness:
                solution.set_genome(new_genome)
                solution.fitness = new_fitness
                solution.forest= new_forest[1]
                solution.x_test= new_forest[2]
                solution.y_test= new_forest[3]
                solution.params= new_forest[4]
                solution.x_train= new_forest[5]
                solution.y_train= new_forest[6]

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
    
    def get_fitness_tree_params(self, genome):
        
        '''get the preformance of a given genome 
        for the ML model of interest-- Decision tree with testing various parameters
        returns: preformance score, the best tree and the testing data associated with it '''
        rand= 0
        crits= ['gini', 'entropy']
        max_depth= [None, 200, 150, 100, 50]
        min_samples= range(2, 100, 10)
        min_samples_leaf= range(1, 50, 10)
        min_weight_fraction_leaf= np.arange(0, 0.5, 0.1)
        max_features= ['auto', 'sqrt', 'log2']
        max_leaf_nodes= [None, 100, 200, 300]
        ccp_alphas= np.arange(0, 2, 0.5)
    
    
        trees= []
        for i in range(len(crits)):
            for j in range(len(max_depth)):
                for k in min_samples:
                    for l in min_samples_leaf:
                        for m in min_weight_fraction_leaf:
                            for n in range(len(max_features)):
                                for o in range(len(max_leaf_nodes)):
                                    for q in ccp_alphas:
                                        tree= create_tree(crits[i], max_depth[j], k, l, m, max_features[n], rand, max_leaf_nodes[o], q)
                                        trees.append(tree)
        
        x_train, x_test, y_train, y_test = train_test_split(self.data[genome], self.data.iloc[:, 226:], test_size = 0.33, random_state = 0)                                
        scores= []
        for t in range(len(trees)):
            tree= trees[t]
            fitted_data= tree.fit(x_train, y_train)
            score= tree.score(x_test, y_test)
            scores.append(score)
        
        high_score= max(scores)
        best_tree_index= scores.index(high_score)
        best_tree= trees[best_tree_index]
    
        return high_score, best_tree, x_test, y_test, x_train, y_train
    
    
    
    def get_fitness_forest_params(self, genome):
        
        '''get the preformance of a given genome 
        for the ML model of interest-- Random Forest with testing various parameters
        returns: preformance score, the forest and the test data used with it'''
        #rand= 0
        crits= ['gini', 'entropy']
        n_estimators= [int(x) for x in np.linspace(start= 200, stop= 3000, num=25)]
        max_features= ['auto', 'sqrt', 'log2']
        max_depth= [int(x) for x in np.linspace(10, 110, num=15)]
        max_depth.append(None)
        min_samples_split= [2, 5, 10, 15, 20]
        min_samples_leaf= [1, 2, 4, 6, 8, 10]
        bootstrap= [True, False]
    
        
        random_grid= {'criterion': crits, 'n_estimators': n_estimators, 'max_features':max_features, 'max_depth':max_depth,
                     'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf, 'bootstrap':bootstrap}
        
        forest= RandomForestClassifier()
        x_train, x_test, y_train, y_test = train_test_split(self.data[genome], self.data.iloc[:, 226:], test_size = 0.33, random_state = 0)
        y_train = y_train.to_numpy().reshape(y_train.shape[0])
        y_test = y_test.to_numpy().reshape(y_test.shape[0])
        
        forest_rand= RandomizedSearchCV(estimator= forest, param_distributions= random_grid, n_iter= 500, cv= 3, verbose= 2, random_state= 42, n_jobs= -1)
        forest_rand.fit(x_train, y_train)
        
        best_forest= forest_rand.best_estimator_
        best_score= best_forest.score(x_test, y_test)
        best_params= forest_rand.best_params_
 
        
        return best_score, best_forest, x_test, y_test, best_params, x_train, y_train


phc10 = ParallelHillClimber(pop_size=2, num_gens=2, cluster_size=10,fitness_file="fitness.csv")
phc20 = ParallelHillClimber(pop_size=2, num_gens=2, cluster_size=20,fitness_file="fitness.csv")
phc25 = ParallelHillClimber(pop_size=2, num_gens=2, cluster_size=25,fitness_file="fitness.csv")
phc30 = ParallelHillClimber(pop_size=2, num_gens=2, cluster_size=30,fitness_file="fitness.csv")
#best = phc.evolve()

besttree10= phc10.evolve_dtree_params()
#besttree20= phc20.evolve_dtree_params()
#besttree25= phc25.evolve_dtree_params()
#besttree30= phc30.evolve_dtree_params()

#bestforest10= phc10.evolve_rforest_params()
#bestforest20= phc20.evolve_rforest_params()
bestforest25= phc25.evolve_rforest_params()
#bestforest30= phc30.evolve_rforest_params()

besttree10

bestforest25

besttree10params= besttree10[4]
#besttree20params= besttree20[4]
#besttree25params= besttree25[4] 
#besttree30params= besttree30[4]
#besttree10params

#bestforest10params= bestforest10[4]
#bestforest20params= bestforest20[4]
bestforest25params= bestforest25[4] 
#bestforest30params= bestforest30[4]
bestforest25params

best_tree= besttree10[1]
best_forest= bestforest25[1]

# +
#view these, note which ones they are, will want table for what they all mean

feat_names10tree= besttree10[5].columns
#feat_names20tree= besttree20[5].columns
#feat_names25tree= besttree25[5].columns
#feat_names30tree= besttree30[5].columns

feat_names25forest= bestforest25[5].columns

# -


feat_names_treelist= list(feat_names10tree)
feat_names_forestlist= list(feat_names25forest)

from sklearn import tree

#make graph of winning decision tree
fig= plt.figure(figsize=(90,50))
a= tree.plot_tree(best_tree, feature_names= feat_names_treelist, filled=True, fontsize='12')
plt.title('Decision Tree', fontsize= 72)
fig.savefig("decistion_tree_1.png")

from sklearn import metrics
import seaborn as sns

x_test= besttree10[5]
x_train= besttree10[7]
y_train= besttree10[8]
y_test= besttree10[6]
y_pred= best_tree.predict(besttree10[5])
y_score= best_tree.predict_proba(besttree10[5])

#make confusion matrix of winning decision tree!
confusion_matrix = metrics.confusion_matrix(besttree10[6], y_pred)

# +
#make confusion matrix of winning decision tree
#confusion_matrix = metrics.confusion_matrix(test_lab, test_pred_decision_tree)

# +

matrix_df= pd.DataFrame(confusion_matrix)
ax = plt.axes()
#sns.set(font_scale=1.3)
plt.figure()
plot= sns.heatmap(matrix_df, annot=True, fmt='g', ax=ax, cmap="magma")
ax.set_title('Confusion Matrix: Decision Tree')
ax.set_xlabel('Predicted label', fontsize=15)
ax.set_xticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=45)
ax.set_ylabel('True Label', fontsize=15)
ax.set_yticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=0)
fig1= plot.get_figure()
#ax.savefig("decistion_tree_1_conf_matrix.png")
fig1.savefig("decistion_tree_1_conf_matrix.png", bbox_inches='tight')

# -

x_testf= bestforest25[5]
#x_trainf= bestforest25[6]
y_trainf= bestforest25[7]
y_testf= bestforest25[6]
y_predf= best_forest.predict(bestforest25[5])
y_scoref= best_forest.predict_proba(bestforest25[5])

#make confusion matrix of winning forest
confusion_matrix_f = metrics.confusion_matrix(bestforest25[6], y_predf)

# +
matrix_df= pd.DataFrame(confusion_matrix_f)
ax = plt.axes()
#sns.set(font_scale=1.3)
plt.figure()
plot= sns.heatmap(matrix_df, annot=True, fmt='g', ax=ax, cmap="magma")
ax.set_title('Confusion Matrix: Decision Tree')
ax.set_xlabel('Predicted label', fontsize=15)
ax.set_xticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=45)
ax.set_ylabel('True Label', fontsize=15)
ax.set_yticklabels(['Nonsmoker', 'Casual Smoker', 'Daily Smoker'], rotation=0)
fig1= plot.get_figure()
#ax.savefig("decistion_tree_1_conf_matrix.png")
fig1.savefig("random_forest_1_conf_matrix.png", bbox_inches='tight')


# -

from sklearn.metrics import RocCurveDisplay

# + endofcell="--"
#roc_auc_score(y_test, y_score, multi_class='ovr')

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

class_of_interest1= label_binarizer.classes_[2] #daily smokers
class_of_interest2= label_binarizer.classes_[1] #casual smokers
class_of_interest3= label_binarizer.classes_[0] #non smokers

class_id1 = np.flatnonzero(label_binarizer.classes_ == class_of_interest1)[0]
class_id2 = np.flatnonzero(label_binarizer.classes_ == class_of_interest2)[0]
class_id3 = np.flatnonzero(label_binarizer.classes_ == class_of_interest3)[0]


# # +


RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id1],
    y_score[:, class_id1],
    name="Daily Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nDaily Smokers vs (Casual and NonSmokers)")
plt.legend()
plt.savefig("ROC_curve_dtree_dailysmokers", bbox_inches='tight')
# -

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id2],
    y_score[:, class_id2],
    name="Casual Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nCasual Smokers vs (Daily and NonSmokers)")
plt.legend()
plt.savefig("ROC_curve_dtree_casualsmokers", bbox_inches='tight')

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id3],
    y_score[:, class_id3],
    name="Non Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nNonSmokers vs (Casual and Daily Smokers)")
plt.legend()
plt.savefig("ROC_curve_dtree_nonsmokers", bbox_inches='tight')

# # +
#also need to get feature weights for each random forest 
# --

# +
#roc_auc_score(y_testf, y_scoref, multi_class='ovr')

label_binarizerf = LabelBinarizer().fit(y_trainf)
y_onehot_testf = label_binarizer.transform(y_testf)
#y_onehot_testf.shape  # (n_samples, n_classes)

class_of_interest1f= label_binarizer.classes_[2] #daily smokers
class_of_interest2f= label_binarizer.classes_[1] #casual smokers
class_of_interest3f= label_binarizer.classes_[0] #non smokers

class_id1f = np.flatnonzero(label_binarizerf.classes_ == class_of_interest1f)[0]
class_id2f = np.flatnonzero(label_binarizerf.classes_ == class_of_interest2f)[0]
class_id3f = np.flatnonzero(label_binarizerf.classes_ == class_of_interest3f)[0]


# # +


RocCurveDisplay.from_predictions(
    y_onehot_testf[:, class_id1],
    y_scoref[:, class_id1f],
    name="Daily Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nDaily Smokers vs (Casual and NonSmokers)")
plt.legend()
plt.savefig("ROC_curve_rforest_dailysmokers", bbox_inches='tight')

RocCurveDisplay.from_predictions(
    y_onehot_testf[:, class_id2],
    y_scoref[:, class_id2],
    name="Casual Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nCasual Smokers vs (Daily and NonSmokers)")
plt.legend()
plt.savefig("ROC_curve_rforest_casualsmokers", bbox_inches='tight')

RocCurveDisplay.from_predictions(
    y_onehot_testf[:, class_id3],
    y_scoref[:, class_id3],
    name="Non Smokers vs rest",
    color="darkorange",
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nNonSmokers vs (Casual and Daily Smokers)")
plt.legend()
plt.savefig("ROC_curve_rforest_nonsmokers", bbox_inches='tight')

# # +
#also need to get feature weights for each random forest 
# -


