#although bagging can be used to reduce the variance
#of the decision tree, the trees we generate in the
#bagging process will be highly corelated

#random forest will constrain the features we use while
#building trees and will force the trees to be different

#decision trees will use the greedy algorithm to find the
#best split point at each step
#this causes them to have high variance if they aren't pruned

#as can be seen in bagging, we reduce the variance by creating
#different trees with randomly chosen subsets of data and combine their predictions

#the problem is that we use the same greedy algorithm while making each
#tree in the bagging process, meaning that similar or the same split points will be chosen
#meaning the trees will be correlated, thus making their predicitons similar

#we can force the trees to be different by limiting the number of features or columns
#the greedy algorithm can evaluate at each split point while creating the tree

#this is called the random forest algorithm
#we force the greedy algorithm to look at only the subset of the features in each step,
#we still use the bagging, meaning we create multiple trees
#but now, each split point will be different among the trees

#for classification problems, we would use the sqrt(number_of_features) to 
#decide how many different features will the algorithm evaluate

#this will cause the trees to be uncorrelated and the predictions to be more diverse

#the difference now is, while calculating the gini index for the best split point
#we will not evaluate all the features but only a randomly chosen subset of the features

#decision trees are great, but they suffer from high variance

#meaning they will produce different trees for different
#training data

#boostrap aggregation of bagging is a method of making
#decision trees more robust and achieve better performance

#the idea is to separate the training data into 
#multiple samples called bootstrap samples

#train multiple trees, one for each sample
#and take their average to produce a final tree
#this reduces the variance and means the tree cannot overfit the problem


#Boostrap resample

#we take random rows from the original dataset
#and add them to a new list
#we can also take, if we choose so, the same row multiple times
#if that same row happens to be chosen multiple times

#the ratio just defines the ratio of the newly created
#dataset and the original dataset

from math import sqrt
from random import seed, randrange
from csv import reader

def subsample(dataset, ratio = 0.1):
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    
    return sample


#we will use k fold cross validation to evaluate the model performance on unseen data
#meaning we will create and evaluate k models and estimate the overall performance
#as their mean error

#we will use the classification accuracy metric, since we will be doing a classification problem
#on the Sonar dataset, we classify the object as a rock or a mine

#we will need all the functions we used in the decision tree project

#we will also need a function, called bagging_precit, to make a predicition
#based on all the outputs of the decision trees, it does so by choosing the most common one

#also, a function called bagging will be used to create the bagging samples of the data,
#train each tree on the sample, and make the prediction based on all the trees


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for g in groups:
        size = float(len(g))

        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = [row[-1] for row in g].count(class_val) / size
            score += p**2

        gini += (1 - score) * (size / n_instances)
    return gini

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score,b_groups = 999,999,999,None
    features = []
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score,b_groups = index,row[index], gini, groups

    return {'index' : b_index, 'value':b_value, 'groups':b_groups}



def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])

    #check for no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    #check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    #process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth,min_size, n_features, depth + 1)

    #process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth,min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root,max_depth,min_size,n_features,1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']  


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def bagging_predict(trees, row):
    predictions = [predict(tree,row) for tree in trees]
    return max(set(predictions), key=predictions.count)



def random_forest(train, test, max_depth, min_size,sample_size,n_trees, n_features):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

seed(2)

filename = 'EnsembleAlgorithms/sonar.all-data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

str_column_to_int(dataset, len(dataset[0]) - 1)

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0]) - 1))

for n_trees in [1,5,10]:
    scores = evaluate_algorithm(dataset,random_forest,n_folds,max_depth,min_size,sample_size,n_trees, n_features)
    print(f"Using {n_trees} trees")
    print(f"Scores {scores}")
    print(f"Mean accuracy is {sum(scores) / float(len(scores))}")




