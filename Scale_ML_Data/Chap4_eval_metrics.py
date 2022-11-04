#this is a way of determining how well our model made the predictions
#that it made

#the information gained for eval_metrics are:

#how different data transformations affect our model
#how well different models perform on the same data
#how different configs of the same model perform on the same data


#Classification accuracy

#the ratio of correctly predicted outcomes and all predictions made

from cgitb import lookup
from enum import unique
import enum
from math import sqrt
from cairo import Matrix


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) *100


actual = [0,0,0,0,1,1,1,1]
predicted = [0,0,1,0,1,1,1,0]

#print(accuracy_metric(actual, predicted))

#used mainly in binary classification problems
#more than 2 classes and this losses it's meaning, and we have to use 



#Confusion matrix

#rows represent predicted class values
#cols represent actual class values

def create_confusion_matrixx(actual, predicted):
    unique_classes = set(actual)
    matrix = [list() for x in range(len(unique_classes))]

    for i in range(len(unique_classes)):
        matrix[i] = [0 for x in range(len(unique_classes))]


    lookup = dict()
    for i, value in enumerate(unique_classes):
        lookup[value] = i

    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1

    precision = 0
    precision += sum([row[0] for row in matrix])

    precision = (1 / precision) * matrix[0][0]

    recall = (1 / sum(matrix[0])) * matrix[0][0] #also called True Positive Rate




    f1_score = 2 / ((1 / precision) + (1 / recall))

    return unique_classes, matrix, precision, recall, f1_score



def print_pretty_confusion_matrix(unique, matrix):
    print(f"Unique classes {[str(x) for x in unique]}")
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


unique_class, matrix, precision, recall, f1 = create_confusion_matrixx(actual, predicted)

print_pretty_confusion_matrix(unique_class, matrix)

print(f"Precision {precision}, Recall {recall}, F1 score {f1}")

#main diagonal says we have 6 correct guesses in total, 3 for each class
#pseudo main says we have 2 incorrect in total,
#once the model said 1 when it was 0, and once it said 0 when it was 1




#MAE, mean absolute error

#used in regression, where we guess continious values

def mae_metric(actual, pred):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(actual[i] - pred[i])
    return sum_error / float(len(actual))

actual = [0.1,0.15,0.2,0.25,0.35]
pred = [0.11,0.19,0.29,0.41,0.5]


def r2_score(actual, pred):
    sum_of_squared_errors = 0
    for i, x in enumerate(pred):
        sum_of_squared_errors = (x - actual[i])**2    
    mean = sum(actual) / float(len(actual))
    sum_of_errors = 0
    for y in actual:
        sum_of_errors += (y - mean) * (y - mean)
    return 1 - (sum_of_squared_errors / sum_of_errors)


print(mae_metric(actual, pred))
print(f"R2 score is {r2_score(actual, pred)}")

#RMSE or Root Mean Squared error

#penalizes larger errors more because of the squaring
#uses rooting to convert the data back into the same units
#for comparison

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        pred_error = predicted[i] - actual[i]
        sum_error += (pred_error**2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


print(rmse_metric(actual, pred))

#RMSE is bigger that MAE, precisely because it penalizes
#larger errors heavier



     





