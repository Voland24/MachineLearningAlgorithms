#similar to PCA, we want to transform the data
#linear discriminant analysis wants to maximize the separability of the data
#i.e. maximizes the separability between the known categories

#it still reduces the dimensionality, but unlike PCA where we reduce the dimensionality 
#by choosing those vectors of variation that explain the major directions of variability change
#here we reduce the dimensionality by maximizing the separability of the known categories of data

#LDA reduces dimensionality by creating a new axis and projects the data onto it
#how? it keeps in mind two criteria simultaneously

#1) it tries to maximize the distance between the means of the categories in the dataset
#2) we also want to minimize variation (or scatter s^2) within each categorie given

# (mean1 - mean2)**2 / (var1**2 + var2**2)
# we square the numerator for the sign
#ideally, we want this ratio to be (a large number) / (a small number)
#i.e. more distance between the means and keep the overall variation as low as possible

#if we had more than 2 categories?
# we would calculate distance as 
#we first find the overall mean of the whole dataset, a central point
#and then calculate the distance to the central point from each of the mean of each category

#so we'd have (cpoint - mean1)**2 + ... + (cpoint - mean_k)**2 / (var1**2 + ... + vark**2)

#secondly, if we had 3 points, LDA would create 2 new axis, because 3 mean points create a plane


#we calculate the within class scatter
#i.e. how much scatter is there per each class

# Swc = sum(si)
#si = sum( (xic - xc_mean) ** 2)), meaning we
#calculate the covariance per class and add it together for all classes


#we also calculate between class scatter
# Sbc = sum( n_labels_of_ci * (x_central_point - xc_mean)**2)
#so number of labels class i * distance of the mean of class i from the overall central point



#calculate Swc^-1 * Sbc
#finds it eigenvector and values
#sort the eigenvalues in decreasing order
#pick first k eigenvectors, those are the linear discriminants
#transform the original data using these linear discriminants


import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.lin_discriminants = None

    def fit(self,X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_overall = np.mean(X, axis=0)
        print(mean_overall)
        SW = np.zeros((n_features,n_features))
        SB = np.zeros((n_features,n_features)) 
        
        for c in class_labels:
            X_c = X[y==c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features,1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SW)

        eigenvalues, eigenvectors = np.linalg.eig(A)

        eigenvectors = eigenvectors.T
        indxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[indxs]
        eigenvectors = eigenvectors[indxs]

        self.lin_discriminants = eigenvectors[0: self.n_components]

    def transform(self,X):
        return np.dot(X, self.lin_discriminants.T)


import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
X, y = data.data, data.target

lda = LDA(2)
lda.fit(X,y)
X_projected = lda.transform(X)

print(f"Shape of X {X.shape}")
print(f"Shape of X after transform {X_projected.shape}")

x1, x2 = X_projected[:, 0], X_projected[:, 1]

plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.colorbar()
plt.show()