#we are going to use the SWedish auto insurance dataset
# 63 observations, 1 input variable and 1 output
#regression problem

#here we are implementing simple linear regression
# y = b0 + b1x, estimating b0 and b1 so that our cost 
#function is minimized

#baseline model has around 72% accuracy RSME cost

#Calculate mean and variance

from math import sqrt
from random import randrange, seed

def mean(values):
    return sum(values) / float(len(values)) if len(values) != 0 else 0.0

def variance(values, mean):
    return sum((x - mean)**2 for x in values)

def covariance(x, x_mean, y, y_mean):
    covar = 0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (y[i] - y_mean)
    return covar

def calculate_coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x,x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

def train_test_split(dataset, split = 0.6):
    train = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy    

def rmse_metric(actual, prediction):
    sum_error = 0
    for i in range(len(actual)):
        prediction_error = prediction[i] - actual[i]
        sum_error += (prediction_error**2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)     

def read_txt(filename):
    dataset = []
    with open(filename, 'r') as file:
        while(True):
            line = file.readline()
            if not line:
                break
            if line[0] == 'X':
                continue
            line = line.strip().split('\t')
            line[1] = line[1].replace(',','.')
            line = list(map(lambda x: float(x), line))
            dataset.append(line)
    return dataset

def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = []
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse

def simple_linear_regression(train, test):
    predictions = []
    b0, b1 = calculate_coefficients(train)
    for row in test:
        pred = b0 + b1*row[0]
        predictions.append(pred)
    return predictions



seed(1)
filename = 'LinearAlgorithms/insurance.txt'
dataset = read_txt(filename)
split = 0.6
rmse = evaluate_algorithm(dataset, simple_linear_regression,split)
print(f"RMSE {rmse}")
            