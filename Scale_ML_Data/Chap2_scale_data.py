# often times it's best to normalize data
#ie get it in range of 0-1, helps calculations 
#and gets all data to be in the same range

#first find min and max of the dataset

from math import sqrt


dataset = [
    [0,0,0,0],
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]

def data_min_max(dataset):
    minmax = []
    for i in range(len(dataset)):
        col_vals = [row[i] for row in dataset]
        min_val= min(col_vals)
        max_val = max(col_vals)
        minmax.append([min_val, max_val])

    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


normalize_dataset(dataset, data_min_max(dataset))
#print(dataset)


#besides normalization, we can also use standardization
#we center the data around the value of 0 and the std of 1
#this is used if the data in question can follow a gaussian distribution



def means_of_cols(dataset):
    means = [0 for _ in range(len(dataset))]
    for i in range(len(dataset)):
        col_vals = [row[i] for row in dataset]
        means[i] = sum(col_vals) / float(len(dataset))
    return means


#print(means_of_cols(dataset))


def col_stdev(dataset, means):
    stdevs = [0 for _ in range(len(dataset))]
    for i in range(len(dataset)):
        variance = [(row[i] - means[i])**2 for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(dataset) - 1))) for x in stdevs]
    return stdevs


#print(col_stdev(dataset, means_of_cols(dataset)))


def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


means = means_of_cols(dataset)
stdevs = col_stdev(dataset, means)
standardize_dataset(dataset, means, stdevs)
print(dataset)

