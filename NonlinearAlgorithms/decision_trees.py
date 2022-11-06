#powerful and popular
#easy to understand by everyone

#foundation for bagging, random forests and gradient boosting

#we will make a CART for binary classifiaction
#it's a literal binary tree, 0,1 or 2 children per node

#creating a binary desision tree is actually 
#dividing up the input space

#recursive binary splitting, numerical procedure
#we line up all the values and try and test all split points
#using a cost function

#we minimize the cost function and thus choose
#the best possible split points

#regression cost: sum squared error across all train examples
#classification: gini cost function tests purity of node
#where purity means how mixed up the training data assigned to each node is

#splitting continues until nodes contain a minimum number of training examples
#or until maximum depth is reached


#Gini index is the cost function used to evaluate splits in the dataset

#it can be used to divide training patterns into two groups of rows

#gini scores tell us how good a split is by how mixed 
#the classes are in the two created groups

#perfect score is 0 because that gives the perfect separation
#worst score means that we have a split where the classes are 
#split 50/50 in each group and the score is 0.5
#for a 2 class problem

#suppose we have 2 classes, 0 and 1
#we divide them into two groups,
#each groups has two rows and both are of the same class
#i.e group 1 has 2 rows of class 0 and group 2 2 rows class 1

#we calculate the proportion of each class in each group
# prop = count(class_examples) / count(rows)

#gini is calculated per child node

# gini = 1 - (sum(prop**2)) * (group_size/total_samples)
#total samples are all examples in the parent node

#then we add up all the scores per child node and get the
#gini score for the split point

from random import seed, randrange
from csv import reader


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


#splittin a dataset means choosing an attribute to split over
#and a value for the attribute which we'll use to split the dataset

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


#we will now use a greedy algorithm to choose the best split
#we will check every value of each attribute, except the class
#and split there
#we will store the best gini scores (Lowest) and thus choose the best 
#possible split

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score,b_groups = 999,999,999,None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score,b_groups = index,row[index], gini, groups

    return {'index' : b_index, 'value':b_value, 'groups':b_groups}



#we must choose when to stop adding new nodes to tree
#after root node

#one is maximum tree depth
#after that number of nodes from root, tree is likely to
#overfit since it's much more comples

#other is minimum node record
#each node is responsible for a given number of training patterns
#once at or below this minimum, we must stop splitting
#adding new nodes. too few training patterns will be too specific 
#and likely will cause overfitting


#predictions are made at terminal nodes, by choosing the
#most common class in the given terminal node

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
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
        node['left'] = get_split(left)
        split(node['left'], max_depth,min_size, depth + 1)

    #process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth,min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root,max_depth,min_size,1)
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



def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            row = list(map(lambda x: float(x.strip()), row))
            dataset.append(row)
    return dataset

# Split a dataset into k folds
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


# Evaluate an algorithm using a cross validation split
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


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = []
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


seed(1)

filename = 'NonlinearAlgorithms/data_banknote_authentication.csv'
dataset = load_csv(filename)

n_folds = 5
max_depth = 5
min_size = 10

scores = evaluate_algorithm(dataset, decision_tree,n_folds,max_depth, min_size)

print(f'Scores {scores}')
print(f"Mean accuracy {sum(scores) / float(len(scores))}")




