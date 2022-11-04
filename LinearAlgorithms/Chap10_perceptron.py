#simples neural network

#single neuron can be used for binary classification



# activation = bias + sum(weights * x)

#pred = 1 if activation >= 0 else 0

#can only be used for linearly separable problems

# w = w + learning_rate * (expected - pred) * x

from cgitb import lookup
from random import randrange, seed


def read_file_csv(filename):
    dataset = []
    unique_classes = set()
    with open(filename, 'r') as file:
        while(True):
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                # line = line.replace(';', ',')
                line = line.split(',')
                numbers = line[:(len(line) - 1)]
                numbers = list(map(lambda x: float(x), numbers))
                unique_classes.add(line[-1])
                numbers.append(line[-1])
                #new_line = numbers.append(line[-1])
                dataset.append(numbers)
    lookup = dict()
    for i, c in enumerate(unique_classes):
        lookup[c] = i

    for row in dataset:
        row[-1] = lookup[row[-1]]

    return dataset, lookup


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i+1] * row[i]

    return 1.0 if activation >=0.0 else 0.0


def train_weights(train, learning_rate, n_epoch, batch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        averages_updates = [0 for _ in range(len(train[0]))]
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error **2
            if not batch:
                weights[0] = weights[0] + learning_rate * error
                for i in range(len(row) - 1):
                    weights[i+1] = weights[i+1] + learning_rate * error * row[i]
            else:
                averages_updates[0] += learning_rate * error
                for i in range(len(row) - 1):
                    averages_updates[i+1] += learning_rate * error * row[i]

        if batch:
            averages_updates = list(map(lambda x: x / float(len(train)), averages_updates))   
            weights = [w + averages_updates[i] for i, w in enumerate(weights)]

        print(f"Epoch {epoch} / {n_epoch}, learning_rate {learning_rate}, sum_error {sum_error}")

    return weights


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


def perceptron(train, test, learning_rate, n_epoch, batch = False):
    predictions = []
    weights = train_weights(train, learning_rate, n_epoch, batch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


seed(1)

dataset, lookup = read_file_csv('LinearAlgorithms/sonar.all-data')

n_folds = 3
learning_rate = 0.01
n_epoch = 500
scores = evaluate_algorithm(dataset, perceptron,n_folds,accuracy_metric,learning_rate,n_epoch, False)

print(f"Scores {scores}")
print(f"Mean accuracy {sum(scores) / float(len(scores))}")


