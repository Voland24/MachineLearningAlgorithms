#we are doing svd basically, but a statistical interpretation of it
#it's going to give us a hierarchical coord system
#that we are going to use to describe the variation in our dataset

#i.e the direction of variation in data that explain the biggest part of the variance in our data


#each row of the data matrix represents a single measurement we have

#firstly, we compute the row-wise mean of the data matrix
#the we create a mean matrix i.e. [1,1....1]^T * [row1_mean, row2_mean, ....., rown_mean]

#then we subtract the mean matrix from the original data matrix. why?
#we do this do mean-center our data, i.e center of distribution is a the origin of the space of the data matrix
#why? because we assume that our data matrix is from a zero mean gaussian distribution

#the we calculate the covariance matrix, by taking the mean center matrix and multiplying it by itself

#if we do SVD of this covariance matrix, we get B*V
#B are the principal componnets i.e U*SIgma from the classi SVD
#V are called loadings and represents the same as V in classic SVD

#what we get are new set of vectors to describe the data
#such that they are orthogonal and ranked by the variance of the data among them i.e. their direction

#so we get linearly independent feature vectors
#we can reduce the dimensionality just by taking those of the highest importance without significant data loss
#newly found dimensions should reduce projection error
#we project the data along the axis (feature vectors) and the spread should have maximum variance

#projection error means that the rmse for example of the original data and the data projected along one or more feature axis should be minimal


#Steps
#calculate mean of X
#calculate covaraince of X,X
#find eigen vectors and values of the covariance matrix
#sort them by eigenvalues in decreasing order
#choose first k vectors to from k dim space
#project the original n space matrix to this new k space

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:
    def __init__(self, n_prin_comps):
        self.n_prin_comps = n_prin_comps
        self.comps = None
        self.mean = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        #calc covariance, transpose because of the numpy function docs
        cov = np.cov(X.T)
        #calc eigenvectors/values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #sort eigenvalues
        eigenvectors = eigenvectors.T
        inds = np.argsort(eigenvalues)[::-1] #decreasing order
        eigenvalues = eigenvalues[inds]
        eigenvectors = eigenvectors[inds]
        #store the first n_prin_comps vectors
        self.comps = eigenvectors[0:self.n_prin_comps]

    def transform(self,X):
        #project data
        X = X - self.mean
        return np.dot(X, self.comps.T)





data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print(f"Shape of X is {X.shape}")
print(f"After transform, shape of X {X_projected.shape}")

x1 = X_projected[:,0]
x2 = X_projected[:,1]

plt.scatter(x1,x2, c=y,edgecolor="none", alpha=0.8,cmap=plt.cm.get_cmap("viridis",3))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

