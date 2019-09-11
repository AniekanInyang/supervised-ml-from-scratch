from util import pre_process_data
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
import numpy as np
from datetime import datetime



class Perceptron(object):
    def fit(self, X, Y, learning_rate=0.001, epochs=1000):
        D = X.shape[1]
        #number of features
        self.w = np.random.randn(D)
        #randomly initialise weights that are the magnitude of the features
        self.b = 0
        #initialise bias to 0

        N = len(Y)
        costs = []

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            #these are incorrect predictions
            if len(incorrect) == 0:
            #no incorrect predictions anymore
                break
            
            #choose a random incorrect sample/prediction
            i = np.random.choice(incorrect)
            self.w = self.w + learning_rate*Y[i]*X[i]
            self.b = self.b + learning_rate*Y[i]

            c = len(incorrect)/float(N)
            #To know how many incorrect samples left
            costs.append(c)
        print ('final w:', self.w, 'final bias:', self.b, 'epochs:', epoch + 1, '/', epochs)

        plt.plot(costs)
        #to see progression of incorrect predictions
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y==P)

if __name__ == '__main__':
    print("Perceptron Model")
    p_data = pre_process_data('loan_data.csv')
    print(p_data)
    p_data = p_data.values

    X = p_data[1:, :-1]
    Y = p_data[1:, -1]

    total_N = len(Y) // 2
    print(total_N)
    
    Xtrain, Ytrain = X[:total_N], Y[:total_N]
    
    Xtest, Ytest = X[total_N:], Y[total_N:]    


    
    model = Perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print ("Time to train model:", (datetime.now() - t0))

    t0 = datetime.now()
    print ("Train accuracy:", model.score(Xtrain, Ytrain))
    print ("Time to compute train accuracy:", (datetime.now() - t0), "Size of train data:", len(Ytrain))

    t0 = datetime.now()
    print ("Test accuracy:", model.score(Xtest, Ytest))
    print ("Tinme to compute test accuracy:", (datetime.now() - t0), "Size of test data:", len(Ytest))




