#we don't know which algorithm will work best
#so what we do is make a test harness and use it
#to evaluate our different models

#Test harnesses consist of 3 parts:
# resampling method to split up dataset
# machine learning model to be evaluated
# performance measure to evaluate predictions


#We are going to use Pima Indian dataset, used for 
#predicting the onset of diabetes within 5 years
#baseline performance is 65%

#it is a binary classification problem
#not a balanced problem in terms of classes observations
# 768 observations with 8 features and 1 output
#missing ones are filled with 0

#Test/train split harness

from random import seed, randrange
from csv import reader

def train_test_split(dataset, split = 0.6):
    train = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy    

def cross_validation_split(dataset, n_folds = 3):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
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

    return correct / float(len(actual)) *100


def zero_rule_algorithm(train, test):
    output_values = [row[-1] for row in train]
    pred = max(set(output_values), key=output_values.count)
    predictions = [pred for _ in test]

    return predictions



def load_csv(filename):
    dataset = []
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


#args used for additional args for algorithm if needed
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = []
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None #to prevent cheating
        test_set.append(row_copy)

    predicted = algorithm(train, test, *args)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy


seed(1)

filename = 'LinearAlgorithms/pima-indians-diabetes.data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

split = 0.6 #standard
accuracy = evaluate_algorithm(dataset, zero_rule_algorithm,split)
print(f" Train/test split Acc achieved is {accuracy}")


#Cross validation harness

#creates k-folds od data, and trains k models,
#one for each combination

def evaluate_algorithm_cross_validation(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, []) #linearization of array of array into 1d array
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


scores = evaluate_algorithm_cross_validation(dataset, zero_rule_algorithm,n_folds=5)
print(f"Scores : {scores}")
print(f"Mean accuracy : {sum(scores) / float(len(scores))}")

#Additional,
# pass in the evalution function, and open the door for regression harnesses
# pass in the function to resample, choose between train/test and k-fold
# calculate std for scores to see how the model is evaluating during k fold





