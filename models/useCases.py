import numpy as np
from linearRegression import linearRegression
from multiRegression import multiRegression
from navieKMeansClustering import naiveKMeansClustering
from simpleLogisticRegression import simpleLogisticRegression
def main():

    #################################
    ##       Linear Regression     ##
    #################################

    model = linearRegression(np.array([1,2,3,4]),np.array([2,4,6,8]))
    model.train(10000,0.01)
    print("Linear Regression Predicted value for 10: ",model.predict(10))

    #################################
    ##       Multi Regression      ##
    #################################

    X = np.array([[1, 2],  
                  [2, 3],  
                  [3, 4],  
                  [4, 5]])  
    y = np.array([3, 5, 7, 9])
    model = multiRegression(X, y)
    model.train(10000, 0.01)  # Train the model
    print("Multi-Regression Predicted value for [5, 6]:", model.predict(np.array([[5, 6]])))

    #########################################
    ##       Naive K-Means Clustering      ##
    #########################################

    data = np.array([1,2,3,11,12,13,21,22,23])
    model = naiveKMeansClustering(data, 3)
    model.train(100)
    print("Naive K-Means Clustering Predicted centroids are: ", model.get_centroids())
    print("Naive K-Means Clustering Predicted centroid for 14: ", model.predict(10))


    #########################################
    ##       Simple Logistic Regression    ##
    #########################################

    X = np.array([[1,2],
                [1,2.5],
                [1,1.5],
                [2,2],
                [10,12],
                [10,20],
                [20,30] ])
    
    y = np.array([1,1,1,1,0,0,0])

    model = simpleLogisticRegression(X,y)
    model.train(1000,0.1)
    print(model.w,model.b)
    print(model.predict([10,20]))


if __name__ == "__main__":
    main()

    
