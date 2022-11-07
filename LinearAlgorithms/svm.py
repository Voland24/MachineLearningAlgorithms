#it's roots lie in the bias/variance tradeoff

#maximum margin classifiers
#pick 2 closes points in clusters, put a split in their
#midway point
#this has very low bias, i.e. it's highly sensitive to new data
#and has high variance i.e it performs very poorly on new test data
#meaning it overfits on the training dataset

#if we allow for misclassifications to exist, and pick some other 2 points from the 2 clusters
#to determine the margin, it's called a soft margin

#this soft margin has higher bias, meaning it's not swayed by every new piece of data
#and lower variance, meaning it performs better on unseen test data

#how to pick the 2 points from each cluster to find the best soft margin?
#we use cross validation to pick the best margin for classifiaction
#meaning we determine how many examples we should allow to be
# missclassified to pick the best spot for the soft margin

#when the data is 2D the Support vector classifier (soft margin)
#forms a line to make a split between the data for classification
#when it's 3D, it forms a plane, etc

#this is great so far, however SUpport vector classifiers aka soft margins won't work if the data has high overlap
#meaning for example
# we are modeling if patients are cured or not based on how much dosage of drug A they received
# patients with low and high dosages weren't cured, but those in some midrange were
#now, no matter where we put the soft margin, there are a lot of mistakes

#this is solved using support vector machines
#the idea is to move the data from a low dimensional space e.g 1D to a higher one e.g 2D
#we would do this by squaring each datapoint in the 1D space and plotting them on a y axis
#then we would use a support vector classifier aka soft margin to determine a hyperplane that is going to
#split the data in this higher dimensional space and be used for classification

#question now is, how did we know to choose dosage**2 to transform the data into 2D?
#why not choose some other function?

#SVM use kernel functions to find support vector classifiers in higher dimensions

#NOTE: Kernels don't actually transform the data into higher dimensions, they just find the relationship between datapoints
# as IF they were in higher dimensions
#this is called the Kernel trick
#this avoids heavy computation required to actually transform the data
#also makes the rbf kernels possible, because it works in infinite dimensions

#one kernel to use would be a polynomial kernel, which finds a support vector classifier in the specified dimensions, d = 2,3,..
#we can use cross validation to find a good value for d

#other popular kernel is radial basis function kernel, aka rbf kernel which works in infinite dimensions
#given new datapoints, it acts as a weighted nearest neighbour and takes into account more the classes of the
#points closer to our new datapoint


#the polynomial kernel looks like (a*b + r)^d, where r is the coefficient and d the dimension

#a and b are the two datapoints we are trying to calclulate the relationship of
#so for r = 1/2 and d=2 i.e. we are working in 2D we have
# (a*b + 1/2) * (a*b + 1/2)
# a^2*b^2 + a*b + 1/4 i.e a*b + a*b + a^2*b^2 + 1/4 and we can represent that as
#  (a,a^2,1/2) dot (b,b^2,1/2) and this reads as 
# a and b are the coordinates of the 2 points in 1D, a^2 and b^2 are the respective coords in 2D
#and 1/2 is the z coord, but since it's the same for both points, we ignore it and stay in 2D

#when r=0, all this kernel does is shift the points on the number line
#(a*b + 0)^d = a^d*b^d which is this dot product (a^d) dot (b^d) and that is still 1D space

#meaning when we want to calculate the relationship between two datapoints in a higher dimension
#we can have to calculate (a*b + r)^d and it will give us the same result as if we actually did the 
#said transformation into higher dimensions


#the rbf kernels is represented as e^(-gamma*(a-b)**2)
#the influence one datapoint a has on datapoint b is a function of the squared distance
#gamma is determined by cross validation is then used to scale the said influence



#what happens when we add multiple polynomial kernels where r = 0 and d = 1,2,3..

# a*b + a^2*b^2 is this dot product (a,a^2) dot (b,b^2) and that is 2D relationship between datapoints i.e. those are the 2D coords for each datapoint
#if we add d = 3, we have (a, a^2,a^3) dot (b,b^2,b^3) and this is 3D coords for each datapoint

#note, we still only calculate (a*b) + (a*b)^2 + (a*b)^3 to actually find the relationship

#what if we keep adding them up until d = +inf?

# a*b + a^2b^2 + .... + a^(+inf) * b^(+inf)

#this is what the radial basis kernel does!

#if the rbf is e^(-gamma(a-b)^2), gamma = 1/2, which is e^(-1/2(a^2 + b^2)) * e^(ab)
#we can do the taylor series expansion of the last term, at a = 0,
#and represent that as the dot product
#that multiply each term in the dot product by sqrt(e^(-1/2(a^2+b^2))) to represent it all as one dot product
#this product has inf dimensions meaning it represents the coords of the points in infD space
#meaning when we calculate the rbf kernel we actually calculate this relationship it infD space
#without actually carrying out the neccessary transformation


#so we are trying to find such a hyperplane
#such that it's margin to datapoints is maximum


#the equation of a hyperplane here is w*x - b = 0
#since this is a 2D problem, binary classification

#so if the w*x - b >=1 it class 1
#and if w*x - b <= -1 it class -1 or 2 or whatever we choose to call it


#together, it means yi*(w*xi - b) >= 1

#so now we must find the weight and bias

#we are going to use hinge loss
# l = max(x, 1 - yi*(w*xi - b))

#also, we are going to regularize the weight by L2 norm
# so we add lambda * || w ||^2

# and the hinge loss is 1/n * sum(max(0, 1 - yi*(w*xi - b)))

import numpy as np

class SVM:
    def __init__(self, lrate = 0.01, lambda_param = 0.01, n_iters = 1000) -> None:
        self.lr = lrate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self,X,y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0,-1,1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                condition = y_[index] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param * self.w - np.dot(x_i,y_[index]))
                    self.b -= self.lr * y_[index]

    def predict(self,X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)



if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)
    # predictions = clf.predict(X)

    print(clf.w, clf.b)

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()




