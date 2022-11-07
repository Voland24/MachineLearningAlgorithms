#we are going to use most similar historical examples
#to classify new data

#we store the whole record
#the when a new piece of data arrives we find
#k most similar examples to it from the dataset
#similar usually means closest in terms of some metric
#e.g euclidean distance
#then, we summarize the results are pick the most common class
#from those k examples

#it can be used for both classification and regression problems


#when doing regression, we take the average value from the nearest neighbours

#distance metric

from csv import reader
from math import sqrt
from random import randrange, seed


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


#next, find k nearest records
#we have to calculate distance from entire dataset in order to do this
#this can get computationally expensive

def get_neighbours(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])

    return neighbours


def predict_classification(train, test_row, k):
    neighbours = get_neighbours(train, test_row, k)
    output_vals = [row[-1] for row in neighbours]
    prediction = max(set(output_vals), key=output_vals.count)
    return prediction


def predict_regression(train, test_row, k):
    neighbours = get_neighbours(train, test_row, k)
    output_vals = [row[-1] for row in neighbours]
    prediction = sum(output_vals) / float(len(output_vals))
    return prediction


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


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        pred_error = predicted[i] - actual[i]
        sum_error += (pred_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error) 


def evaluate_algorithm(dataset, algorithm, metric, n_folds, *args):
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
        accuracy = metric(actual, predicted)
        scores.append(accuracy)
    return scores



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



# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])



def knn(train, test, predict_func, k):
    predictions = []
    for row in test:
        output = predict_func(train, row, k)
        predictions.append(output)
    return predictions


seed(1)
filename = 'NonlinearAlgorithms/abalone.data.csv'
dataset = load_csv(filename)
for i in range(1, len(dataset[0])):
    str_column_to_float(dataset, i)

str_column_to_int(dataset,0)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 5
k = 5
scores = evaluate_algorithm(dataset,knn,rmse_metric,n_folds, predict_regression,k)
print(f"Scores {scores}")
print(f"Mean score {sum(scores) / float(len(scores))}")