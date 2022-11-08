#we want to cluster data into k different clusters
#unsupervised learning problem

#each sample is assigned to the cluster that has its 
#mean nearest to the given sample

#iterative process
#initialize clusters randomly
#repeat until convergence
#assign points to its nearest cluster
#update cluster centroids, i.e set new centroids to be the mean of the newly formed clusters

#we can use euclidean distance for metric, but there are many other choices

from collections import Counter
import numpy as np

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self,X, y):
        self.X_train = X
        self.y_train = y

    def predict(self,x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]        
        k_indxs = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indxs]
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]

    def predict_all(self,X):
        y_pred = [self.predict(x) for x in X]
        return np.array(y_pred)


from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / float(len(y_true))


iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

k = 3
classifier = KNN(k)
classifier.fit(X_train, y_train)
predictions = classifier.predict_all(X_test)
print(f"KNN accuracy is {accuracy(y_test, predictions)}")

