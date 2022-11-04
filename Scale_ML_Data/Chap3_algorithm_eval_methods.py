#the goal is to create models that accurately predict
#the behaviour of the system or the test data

#we don't have access to test data at the time of training
#so we use statistical methods to esitmate performance of new data

#these are called resampling methods, because we are going to resample
#our available training data

#these help us decide on model parameters and the models we're going to use


#TEST and TRAIN SPLIT

#we split the data into a test and train set
#the splitting process is random
#if we were to compare models, we should use the same split sets
#or the same random seed used to split the dataset

#the default split ratio is 60/40 train/test



from random import randrange, seed


def train_test_split(dataset, split = 0.6):
    train = []
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy    

seed(1)
dataset = [[i] for i in range(1,11)]
train, test = train_test_split(dataset)
#print(train)
#print(test)



#K-FOLD cross validation split

# train/test split can be prone to noise and thus
#not generate a good evalution of the performance

#it splits the data into k groups, i.e. folds
#the model is trained and evaluted k times,
#the performance is a mean of the performance on each of the k times

#the model is trained on k-1 folds and tested on 1
#each time the 1 for testing will be a different one

#this leads us to pick such k that the number of rows
#in our dataset must be divisible by k, so each fold
#has the same number of datapoints

#the folds must be representative of the data, so
# k = 3 is good for small dataset, and k=10 for larger ones

#we check if the folds are good enough to be representative
#by finding the mean and the std of the fold and comparing it to
#the mean and std of the whole dataset


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


print(cross_validation_split(dataset, n_folds=4))


#for giant datasets, n = 100k or 1mil, it is better to
#just use test/train since is time consuming to use k folds

#k means is more accurate but it requires k models to be run and trained

#other possible resampling method are:

#Repeated test/train - same as test/train but done multiple times

#Leave-one out k fold, it is where k = 1 constantly

#Stratification - used in classification, each group chosen
#as a split must have the same distrbution of classes as the original dataset




