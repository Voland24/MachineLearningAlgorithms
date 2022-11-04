#baseline models are used as a benchmark for comparison
#further on when we implement more complex models

#these models perform on a certain problem and we can
#evaluate them using RMSE or MAE if the problem is regression

#then we can later use these performance metrics to comment
#whether or not out newer models are better or not than these
#baseline, naive approaches


#Random Prediction ALgorithm

#basically, the algorithm randomly guesses the outputs

#it takes the labels from the training set and randomly 
#guesses the outputs on the test set

from random import randrange,seed


def random_prediction_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = []
    for row in test:
        index = randrange(len(unique))
        predicted.append(unique[index])

    return predicted

seed(1)
train = [[0],[1],[0],[1],[0],[1]]
test = [[None], [None], [None]]

print(random_prediction_algorithm(train, test))



#A better choice would be the Zero Rule Algorithm
#is creates one rule dependant on the given dataset at hand
#and uses this info to make a prediction

#just like the random algorithm, it can be used both on
#classification as well as regression problems

#Zero Rule for classification

#the rule used here is "always guess" the most common 
#class in the dataset

def zero_rule_algorithm(train, test):
    output_values = [row[-1] for row in train]
    pred = max(set(output_values), key=output_values.count)
    predictions = [pred for _ in test]

    return predictions

train = [[0],[0],[0],[0],[1],[1]]
test = [[None],[None],[None]]

print(zero_rule_algorithm(train, test))

#Zero Rule for Regression

#here, we are guessing a continious value
#the best zero rule guess would be a central tendency
#such as the mean or the median

def zero_rule_regression(train, test):
    output_values = [row[-1] for row in train]
    pred = sum(output_values) / float(len(output_values))
    predicted = [pred for _ in test]

    return predicted


train = [[10],[23],[5],[7],[0]]
test = [[None], [None], [None]]

print(zero_rule_regression(train, test))


