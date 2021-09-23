import random
import numpy as np 
import math

#Q1
def import_data(filename):
    with open(filename, 'r') as f:
        match_result = f.read().splitlines()
    X = []
    Y = []
    #iterating every number in 2d list
    for x in match_result:
        #print(x)
        temp = []
        X1 = x.split(',')
        for id in X1:
            if id == '?':
                temp += ["NaN"]
            else:
                temp += [float(id)]
        X += [X1[-1]]
        temp = temp[:-1]
        Y += [temp]
            
    return  X,Y

#find median
#Q2a
def median(ls):
    sortLst = sorted(ls)
    lstLen = len(ls)
    mid = (lstLen - 1) // 2
    if (lstLen % 2):
        return sortLst[mid]
    else:
        return (sortLst[mid] + sortLst[mid + 1])/2.0

#For every list, remove "NaN" and compute median, insert median at end of the list
def imputemissing1(X):
    
    #every row
    X = [i for i in X if i != "NaN"]
    
    X = sorted(X)
    
    X += [median(X)]
    return X

#iterating every list in X
def imputemissing(X):
    for i in range(len(X)):
        X[i] = imputemissing1(X[i])
    return X
#2b: Because the data contains many large/small variables, and the mean is being skewed by those numbers.
#2c
#return X as origin list, y is the list after discard
def discard_missing(X,y):
    y1 = []
    for i in range(len(y)):
        
        if "NaN" in y[i]:
            continue
        else:
            y1 += y[i]

            
    return X,y1

#3a
#shuffle data randomly
def shuffle_data(X,y):
    temp = list(zip(X, y))
    random.shuffle(temp)
    X, y = zip(*temp)

    return X,y

#3b
#calculating std
def std(X):
    mean = sum(X)/len(X)
    
    return (sum((x - mean) ** 2 for x in X) / (len(X) - 1))**(1/2)

#calculating std in 2d list
def compute_std(X):
    for i in range(len(X)):
        #print(i)
        X[i] = std(X[i])
    return X

#3c
#remove entries with 2 std away from the mean in a list
def remove_outlier1(X):
    #print(X)
    mean = sum(X)/len(X)
    stdv = std(X)
    upperBound = mean + 2 * stdv
    #print(upperBound)
    lowerBound = mean - 2 * stdv
    #print(mean)
    #print(lowerBound)
    for i in X:
        if(i > upperBound) or (i < lowerBound):
            X.remove(i)
    return X

#remove entries with 2 std away from the mean in lists(X: 2d list, Y: 1d list)
#return X,y without outlier
def remove_outlier(X,y):
    for i in range(len(X)):
        X[i] = remove_outlier1(X[i])
    
    y = remove_outlier1(y)
    return X,y

#3d
#standardize 1d list
def standardize_1dlist(X):
    mean = sum(X)/len(X)
    stdv = std(X)
    for x in range(len(X)):
        X[x] = (X[x] - mean) / stdv
    return(X)

#standardize 2d list

#Time and space complexity are both O(n^2)

def standardize_data(X):
    for x in range(len(X)):
        X[x] = standardize_1dlist(X[x])

    return X

#4
#X = All non-numerical values, y = survive

#example from class
def read_triangle(path_to_file):
    # input in a string representing the file location
    # returns a list of list of ints

    f = open(path_to_file, 'r')
    lines = f.readlines()
    triangle = []
    for line in lines:
        triangle.append([int(x) for x in line.split(' ')])
    f.close()
    return triangle

    
def import_non_num_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    match_result = []
    for line in lines:
        match_result.append([x for x in line.strip().split(' ')])
    X = []
    Y = []
    #iterating every number in 2d list
    for x in match_result:
        #print(x)
        temp = []
        X1 = x
        #print(X1)
        for id in X1:
            if id == "female":
                temp += [0]
            elif id == "male":
                temp += [1]
            elif id == 'C':
                temp += [0]
            elif id == 'Q':
                temp += [1]
            elif id == 'S':
                temp += [2]
            else:
                try: 
                    float(id)
                    temp += [float(id)]
                except ValueError:  
                    continue

        if len(X1) > 1:        
            X += [temp]

            Y += [X1[1]]
            
    return  X,Y

            

#5a
#
def train_test_split( X,y,t_f):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    temp = np.random.choice(len(X) - 1, math.floor(len(X)*t_f), replace=False)
    temp2 = np.random.choice(len(y) - 1, math.floor(len(y)*t_f), replace=False)
    print(temp2,temp)
    
    for i in range(len(temp)-1):
       
        X_train += X[temp[i]]
        X.remove(X[temp[i]])
        
    for i in range(len(temp2)-1):
      
        y_train += y[temp2[i]]
        y.remove(y[temp2[i]])
    return X_train, X,y_train,y

#5b
def train_test_split( X,y,t_f, cv_f):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_cvf = []
    y_cvf = []
    
    temp = np.random.choice(len(X) - 1, math.floor(len(X)*t_f), replace=False)
    temp2 = np.random.choice(len(y) - 1, math.floor(len(y)*t_f), replace=False)
    
    
    for i in range(len(temp)-1):
       
        X_train += X[temp[i]]
        X.remove(X[temp[i]])
        
    for i in range(len(temp2)-1):
      
        y_train += y[temp2[i]]
        y.remove(y[temp2[i]])
    
    X_test = X
    Y_test = Y
    temp = np.random.choice(len(X) - 1, math.floor(len(X)*cv_f), replace=False)
    temp2 = np.random.choice(len(y) - 1, math.floor(len(y)*cv_f), replace=False)
    
    for i in range(len(temp)-1):
       
        X_cvf += X[temp[i]]
        X.remove(X[temp[i]])
        
    for i in range(len(temp2)-1):
      
        y_cvf += y[temp2[i]]
        y.remove(y[temp2[i]])
    return X_train, X_test,y_train,y_test,X_cvf,y_cvf

            



#A = [[75,0,190,80],[91,193,371,174,12110],[0,9,9,4]]
#B = [91,193,371,174,999]
#A = standardize_data(A)
#print("here1",A)
#print(import_data("test.txt"))







