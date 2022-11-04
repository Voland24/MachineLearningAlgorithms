#if we have more than one feature in our feature
#vector, the linear regression
#will fit a plane or a hyperplane to out outputs


#its called linear because we model
#the relationship between the inputs and the outputs
#via a linear combination of the inputs
#predictions is done by adjusting the coefficinets
#in this linear combiation

#we will adjust the coefficients using SGD
#we seek to minimize the error (cost function)
#for our model

from random import seed, randrange
from math import sqrt


def preprocess_csv_file(filename):
    dataset = []
    with open(filename, 'r') as file:
        while(True):
            line = file.readline()
            if not line:
                break
            line = line.strip()
            line = line.replace(';', ',')
            numbers = line.split(',')
            numbers = list(map(lambda x: float(x), numbers))
            dataset.append(numbers)        
    return dataset

#fucntion for predicting the output
def predict(row, coefficients):
    y_pred = coefficients[0]
    for i in range(len(row) - 1):
        y_pred += row[i] * coefficients[i + 1]
    return y_pred


#estimating coefficients

#three loops needed for this
#loop over each epoch
#loop over each row in the data in the epoch
#loop over each coefficients and update it

#coefficient of 0, called bias is updated as
# b[0] = b[0] - learning_rate * err, err is pred - truth
#all others are also a function of the input they are weighing
#ie b[i] = b[i] - learning_rate * error * x[i]


def coeffs_optimize_sgd(train, learning_rate, n_epoch, batch = False):
    coef = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        average_update = [0.0 for _ in range(len(coef))]
        sum_error = 0
        for row in train:
            y_pred = predict(row, coef)
            error = y_pred - row[-1]
            sum_error += error **2
            if not batch:
                coef[0] = coef[0] - learning_rate * error
                for i in range(len(row) - 1):
                    coef[i+1] = coef[i+1] - learning_rate * error * row[i]
            else:
                average_update[0] += learning_rate * error
                for i in range(len(row) - 1):
                    average_update[i+1] = learning_rate * error * row[i]

        if batch:
            print(f"Batch update once per epoch!")
            average_update = list(map(lambda x: x / float(len(train)), average_update))
            print(f"AVERAGE UPDATE VECTOR {average_update}")
            for i, avg_update in enumerate(average_update):
                coef[i] = coef[i] - avg_update
        print(f"Epoch {epoch} / {n_epoch}, learning rate {learning_rate}, error {sum_error}")        
    return coef



def linear_regression_sgd(train, test, learning_rate, n_epoch, batch = False):
    predictions = []
    coef = coeffs_optimize_sgd(train, learning_rate, n_epoch, batch)
    for row in test:
        y_pred = predict(row, coef)
        predictions.append(y_pred)
    return predictions    


# dataset = [[i,i] for i in range(1,6)]
# learning_rate = 0.1
# n_epochs = 50
# coef = coeffs_optimize_sgd(dataset, learning_rate, n_epochs)
# print(coef)


#Wine dataset
#using k folds cross-validation, meaning we train k models
#mean error of all models in our performance metric
#we will use rmse error for cost function


#find min and max for each feature
def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        values_min = min(col_values)
        values_max = max(col_values)
        minmax.append([values_min, values_max])
    return minmax    


#normalize dataset
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def means_of_cols(dataset):
    means = [0.0 for _ in range(len(dataset))]
    for i in range(len(dataset[0])):
        col_vals = [row[i] for row in dataset]
        means[i] = sum(col_vals) / float(len(dataset))
    return means

def col_stdev(dataset, means):
    stdevs = [0 for _ in range(len(dataset))]
    for i in range(len(dataset[0])):
        variance = [(row[i] - means[i])**2 for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / float(len(dataset) - 1)) for x in stdevs]
    return stdevs

def standardize_dataset(dataset, means, stdev):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i] / stdev[i])

#split dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / (n_folds))
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


#calculate cost i.e rmse metric
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        predction_error = predicted[i] - actual[i]
        sum_error += predction_error **2 
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


#create a harness for the model
#dataset, type of split, algorithm used, metric used

def evaluate_algorithm(dataset, algorithm, n_folds, metric, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        error = metric(actual, predicted)
        scores.append(error)
    return scores


seed(1)

dataset = preprocess_csv_file("LinearAlgorithms/winequality-white.csv")

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# means = means_of_cols(dataset)
# stdevs = col_stdev(dataset, means)
# standardize_dataset(dataset,means,stdevs)

n_folds = 5
learning_rate = 0.01
n_epochs = 150
scores = evaluate_algorithm(dataset,linear_regression_sgd,n_folds,rmse_metric,learning_rate, n_epochs, True)

print(f"Scores {scores}")
print(f"Mean RMSE: {sum(scores) / float(len(scores))}")




