#or stacking, is a method in which we use a new model
#to best combine predicitons of two or more models trained on our dataset

#this process is often also reffered to as blending

#typically, we use a linear model to combine the predictions
#simple averaging as in voitng, weighted sum using linear reggression or logistic regression
#models whose predictions we combine need to be taught on the problem but needn't be the best possible solutions

#they needn't be fine tuned
#importantly, models need to be uncorrelated, meaning they have to make different predictions
#so models, need to be skilled but differently skilled
#this is achieved by either using completely different models, or models
#that are trained on different representation of data


#We are going to use 2 models, KNN and Perceptron and combine them using logistic regression

#KNN

from math import exp, sqrt
from random import randrange, seed
from csv import reader

def knn_model(train):
    if not train:
        return
    if len(train) == 0:
        return
    
    return train

#KNN uses the whole model, we simply find K most similar datapoints and average them or count the most common class to make a predicition

#we are going to use Euclidean distance

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_neighbours(train, test_row, num_neighbours):
    distances = []
    for train_row in train:
        dist = euclidean_distance(train_row, test_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup : tup[1])
    neighbours = []
    for i in range(num_neighbours):
        neighbours.append(distances[i][0])
    return neighbours

def knn_predict(model, test_row, num_neighbours = 2):
    neighbours = get_neighbours(model,test_row,num_neighbours)
    output_values = [row[-1] for row in neighbours]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


#Perceptron

def perceptron_predict(model, row):
    activation = model[0]
    for i in range(len(row) - 1):
        activation += model[i+1] * row[i]
    return 1.0 if activation >= 0 else 0.0

def perceptron_model(train, l_rate = 0.01, n_epoch = 5000):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            prediction = perceptron_predict(weights, row)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i+1] = weights[i+1] + l_rate * error * row[i]
    return weights

#Aggregator model, Logistic regression

#we use the wights in the logistic reggression to choose the best combination of
#the outputs of out 2 models and get the best predicition possible

def logistic_regression_predict(model, row):
    pred = model[0]
    end = len(row) if len(row) == 61 else len(row) - 2
    for i in range(end - 1):
        pred += model[i+1] * row[i]
    return 1.0 / (1.0 + exp(-pred))

def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
    coef = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            pred = logistic_regression_predict(coef,row)
            error = row[-1] - pred
            coef[0] = coef[0] + l_rate * error * pred * (1 - pred)
            for i in range(len(row) - 1):
                coef[i+1] = coef[i+1] + l_rate * error * pred * (1 - pred) * row[i]
    
    return coef


#we have to train our logistic reggression model on some dataset
#we will construct a dataset that looks like

#KNN Per Truth
#0    0   0
#0    1   1

#in essence, the first two columns are predictions of our given models and the third one is the target or truth

#we will teach the logistic reggression model to best combine these predictions and solve our problem

#sometimes it's a good idea to train the aggregator model on both
#the training row and the predictions made by the submodels
#we do this by aggreagating the training row and the row of predicitons made by the submodels

def to_stacked_row(models,predict_list, row):
    stacked_row = []
    for i in range(len(models)):
        prediction = predict_list[i](models[i], row)
        stacked_row.append(prediction)
    stacked_row.append(row[-1])
    return row[0:len(row) - 1] + stacked_row


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


def stacking(train, test):
    model_list = [knn_model, perceptron_model]
    predict_list = [knn_predict, perceptron_predict]
    models = []
    for i in range(len(model_list)):
        model = model_list[i](train)
        models.append(model)
    stacked_dataset = []
    for row in train:
        stacked_row = to_stacked_row(models,predict_list,row)
        stacked_dataset.append(row)
    stacked_model = logistic_regression_model(stacked_dataset)
    predictions = []
    for row in test:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
        prediction = logistic_regression_predict(stacked_model, stacked_row)
        prediction = round(prediction)
        predictions.append(prediction)
    
    return predictions


seed(1)

filename = 'EnsembleAlgorithms/sonar.all-data.csv'
dataset = load_csv(filename)

for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

str_column_to_int(dataset, len(dataset[0]) - 1)
n_folds = 3
scores = evaluate_algorithm(dataset, stacking,n_folds)
print(f"Scores {scores}")
print(f"Mean accuracy {sum(scores) / float(len(scores))}")

