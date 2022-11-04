#go to algorithm for binary classification

#called linear because the coeff are still
#linerally combined, but used in a exp function

#the outputs are binary, 0 or 1
#it actually gives a real value between 0 and 1
#which is then rouned up to the nearest integer
#the numbers are then mapped to a given class

#y_pred = 1 / (1 + exp(-(b0 + b1*x1)))

#the coeffs are optimized using sgd, like in regression
# b = b + learning_rate * (y - ypred) * ypred * (1 - ypred)*x


#prediction function

from math import exp
from random import seed, randrange
from csv import reader

def predict(row, coefficients):
    y_pred = coefficients[0]
    for i in range(len(row) - 1):
        y_pred += coefficients[i+1] * row[i]
    return 1.0 / (1.0 + exp(-y_pred)) 


#updatin the coeff using sgd
#coeff[0] is not dependant on the x values, it is the intercept


def coefficients_sgd(train, learning_rate, n_epoch, batch = False):
    coef = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        average_update = [0.0 for _ in range(len(train[0]))]
        for row in train:
            y_pred = predict(row, coef)
            error = row[-1] - y_pred
            sum_error += error ** 2
            if not batch:
                coef[0] = coef[0] + learning_rate * error * y_pred * (1 - y_pred)
                for i in range(len(row) - 1):
                    coef[i+1] = coef[i+1] + learning_rate*error* y_pred * (1 - y_pred) * row[i]
            else:
                average_update[0] += learning_rate * error * y_pred * (1 - y_pred)
                for i in range(len(row) - 1):
                    average_update[i+1] += learning_rate*error* y_pred * (1 - y_pred) * row[i]

        if batch:
            average_update = list(map(lambda x: x / len(train), average_update))
            coef = [c + average_update[i] for i,c in enumerate(coef)]
            
        print(f"Epoch {epoch} / {n_epoch}, learning rate {learning_rate}, total_error {sum_error}")

    return coef            


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            row = list(map(lambda x: float(x), row))
            dataset.append(row)
    return dataset



#find min and max for each feature
def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        values_min = min(col_values)
        values_max = max(col_values)
        minmax.append([values_min, values_max])
    return minmax 


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

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


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100


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
        acc = metric(actual, predicted)
        scores.append(acc)
    return scores

def logistic_regression(train, test, learning_rate, n_epoch, batch = False):
    predictions = []
    coefs = coefficients_sgd(train, learning_rate,n_epoch, batch)
    for row in test:
        y_pred = predict(row, coefs)
        y_pred = round(y_pred)
        predictions.append(y_pred)
    return predictions


seed(1)

dataset = load_csv("LinearAlgorithms/pima-indians-diabetes.data.csv")

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 5
learning_rate = 0.1
n_epochs = 200

scores = evaluate_algorithm(dataset,logistic_regression,n_folds, accuracy_metric, learning_rate, n_epochs, False)

print(f"Scores {scores}")

print(f"Mean accuracy {sum(scores) / float(len(scores))}")

