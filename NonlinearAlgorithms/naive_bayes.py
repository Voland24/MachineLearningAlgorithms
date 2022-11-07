#used for binary and multi class classification
#we can calculate if the piece of data belongs 
#to a given class if we have some prior knowledge

#p(class | data) = p(data | class) * p(class) / p(data)

#we assume that the attributes of a given class
#do not interact
#this is highly unlikely in real world data
#however this approach works even then

#firstly, we will separate the dataset by class
#so we can calculate the probability of data 
#given the class they belong to

from math import exp, log, pi, sqrt
from random import seed, randrange

from csv import reader

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)

    return separated


#we also need the mean and the stddev from the dataset

def mean(column):
    return sum(column) / float(len(column))


def stddev(column):
    avg = mean(column)
    variance = sum([(x - avg)**2 for x in column]) / float(len(column) - 1)
    return sqrt(variance)


def summarize_dataset(dataset):
    summaries = [(mean(column), stddev(column), len(column)) for column in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in iter(separated.items()):
        summaries[class_value] = summarize_dataset(rows)

    return summaries


#we assume the values for a real world X1 are drawn
#from a distribution, for example a gaussian one

def calculate_probability(x, mean, stddev):
    exponent = exp(-((x - mean)**2 / (2*stddev**2)))
    return (1 / (sqrt(2*pi) * stddev)) * exponent

#we will now calculate the probability that a new piece of data
#belongs to any of out given classes

# p(class | data) = p(X | class) * p(class)

#Note: this is not strictly a probability.
#we will calculate this for each of our classes 
#from the training set, and select the max value as a prediction
#we aren't really concerned about the actual probability
#hence there is no division in the equation

#for example, for 2 inputs we would have
# p(class = 0 | X1, X2) = p(X1 | class = 0) *
#                         p(X2 | class = 0) * 
#                         p(class = 0)

#we treat the variables as though there is no connection between them


#we calculate p(class = 0) as the number of class = 0 examples
#from our training set divided by the number of all training set rows

#we calculate the p(X1, class = 0) from the gaussian probabilty
#and the mean and stddev from that given column

def calculate_class_probalities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in iter(summaries.items()):
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stddev, count = class_summaries[i]
            probabilities[class_value] *= log(calculate_probability(row[i], mean, stddev) + 1)
    
    return probabilities


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


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


def predict(summaries, row):
    probabilities = calculate_class_probalities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in iter(probabilities.items()):
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value

    return best_label


def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = []
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


filename = 'NonlinearAlgorithms/iris.data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

str_column_to_int(dataset, len(dataset[0]) - 1)

n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes,n_folds)
print(f"Scores {scores}")
print(f"Mean score is {sum(scores) / float(len(scores))}")






