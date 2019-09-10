from util import pre_process_data
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
import numpy as np
from future.utils import iteritems
from datetime import datetime

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        Yhat = np.zeros(len(X))
        #Initialising an array of zeros for the predicted Y
        for i,x in enumerate(X):
        #this is test X data
            sl = SortedList()
            for j, xt in enumerate(self.X):
            #this is train X data
                d = x - xt
                #difference between x in test and train
                diff = d.dot(d)
                #getting the distance, we're using the sum of squares implemented with the numpy dot product
                if len(sl) < k:
                #if length of sorted list is less than k, just add the distance and then the corresponding y in train
                    sl.add((diff, self.Y[j]))
                else:
                    if diff < sl[-1][0]:
                    #else if the distance is less than the last in the array, 
                    #remove the last in the array and add the new distance and correpsonding y
                        del sl[-1]
                        sl.add((diff, self.Y[j]))
        
            votes = {}
            #we store votes in a dictionary
        
            for diff, v in sl:
            #for distance, class in sorted list
                votes[v] = votes.get(v,0) + 1
                #setting class(v) a key to the dictionary, votes.get checks if the key(v) is in the dictionary, if no,
                #set it to 0, then add 1 (this is count). If it exists, add 1 to that count
            max_votes = 0
            max_votes_class = -1
            for v,count in iteritems(votes):
            #for class, count in votes dictionary
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
                #save the class with max count and the count
            Yhat[i] = max_votes_class
            #replace the initialized array of zeroes with the class with the maximum votes for the ith index
        return Yhat
        
    def score(self, X, Y):
        P = self.predict(X)
        #predicted Y
        return np.mean(P==Y)
        #average of the boolean (if Y and P are same)


if __name__ == '__main__':
    print("KNN Model")
    k_data = pre_process_data('loan_data.csv')
    print(k_data)
    k_data = k_data.values

    X = k_data[1:, :-1]
    Y = k_data[1:, -1]

    total_N = len(Y) // 2
    print(total_N)
    
    Xtrain, Ytrain = X[:total_N], Y[:total_N]

    Xtest, Ytest = X[total_N:], Y[total_N:]

    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5,6,7)
    for k in ks:
    #cross validating with different values of k to find the best choice. 
    #k = 5 gave the best result in term of train accuracy and test accuracy
        print ("\nk:", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))
        #timing how long it took to train the model

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Time to get train accuracy:", (datetime.now() - t0), "Length of train data:", len(Ytrain))
        #timing how long it took to get the train accuracy

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        test_scores.append(test_score)
        print("Time to get test accuracy:", (datetime.now() - t0), "Length of test data:", len(Ytest))
        #timing how long it took to get the test accuracy


    print("Train scores:", train_scores)
    print("Test scores:", test_scores)
    #printing the train accuracy and test accuracy values
    #and plotting these accuracies below
    
    plt.plot(ks, train_scores, label="train_scores")
    plt.plot(ks, test_scores, label="test_scores")
    plt.legend()
    plt.show()
    


