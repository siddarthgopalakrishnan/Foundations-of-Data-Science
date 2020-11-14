import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time

def gradient_descent(X, Y, x_test, y_test):

	alpha = 0.0001		# Learning Rate
	epochs = 200000  	# Number of iterations
	m = len(X_train)
	Theta = np.array([1, 1, 1, 1]).reshape(4,1)

	for i in range(epochs):

		#parameter update step
		Theta = Theta - alpha*(1/m)*np.transpose(X)@(X@Theta - Y)

		#print training error value every x iterations
		# if i%20000 == 0:
		# 	cost = (1/(m))*np.transpose((X@Theta - Y))@(X@Theta - Y)
		# 	print("Cost at iteration ", i, " = ", cost.flatten())


	# print("Coefficient matrix from gradient descent= ", Theta)

	#storing the testing error
	m = len(X_test)
	cost = (1/(m))*np.transpose((x_test@Theta - y_test))@(x_test@Theta - y_test)
	return math.sqrt(cost)
###-------------------------- gradient descent done -------------------------------------###


def stochastic_gradient_descent(X, Y, x_test, y_test):

	alpha  = 0.01			#learning rate
	cost = 0				#cost value
	max_epochs = 800000	#max iter
	m = len(X_train)		#training set size
	Theta = np.array([1, 1, 1, 1]).reshape(4,1)

	for i in range(max_epochs):

		zx = X[i%m].reshape(1,-1) #1x4
		zy = Y[i%m].reshape(1,-1) #1x1
	    
	    #parameter update step
		Theta = Theta - alpha*(1/m)*np.transpose(zx)@(zx@Theta - zy)
	    
	    #print training error value every x iterations
		# if i%80000 == 0:
		# 	cost = (1/(m))*np.transpose((X@Theta - Y))@(X@Theta - Y)
		# 	print("Cost at iteration ", i, " = ", cost.flatten())


	# print("Coefficient matrix from stochastic gradient descent= ", Theta)

	#storing the testing error
	m = len(X_test)
	cost = (1/(m))*np.transpose((x_test@Theta - y_test))@(x_test@Theta - y_test)
	return math.sqrt(cost)

###---------------------- stochastic gradient descent done ------------------------------###


def normal_equation_solver(x1, y1, x_test, y_test):

	#main equation:
	#Theta = (XTX)^(-1) . (XTY) 
	
	#Calculate the formula
	x_transpose = np.transpose(x1)   				#calculating transpose
	x_transpose_dot_x = x_transpose.dot(x1) 		#calculating dot product
	temp_1 = np.linalg.inv(x_transpose_dot_x) 		#calculating inverse

	temp_2 = x_transpose.dot(y1)  
	theta = temp_1.dot(temp_2)

	# print("Coefficient matrix after solving normal equations = ", theta)

	m = len(X_test)
	cost = (1/(m))*np.transpose((x_test@theta - y_test))@(x_test@theta - y_test)
	return math.sqrt(cost)
###------------------------- normal equations solved -------------------------------------###

def my_train_test_split(X, y, train_size):
	'''
	my own data splitting function
	'''
	arr_rand = np.random.rand(X.shape[0])
	split = arr_rand < np.percentile(arr_rand, train_size*100)

	X_Train = X[split]
	y_Train = y[split]
	X_Test =  X[~split]
	y_Test = y[~split]

	return X_Train, X_Test, y_Train, y_Test

def standardise_values(a):
	'''
	standardisation function, make data mean  = 0 and sd = 1
	'''
	mean = np.mean(a)
	sd = np.std(a)
	for i in range(len(a)):
		a[i] = (a[i]-mean)/sd
	return a


#General part

#load dataset
dataset = pd.read_csv('insurance.txt')

x = dataset.iloc[:,:-1].values		#feature attributes
y = dataset.iloc[:,-1:].values		#target attribute
m = 1338 							#number of training samples
# print("Shape of x = ", x.shape)		#(1338,3)
# print("Shape of y = ", y.shape)		#(1338,1)


#Standardisation
#standardise feature variables
for i in range(len(x.T)):
	x.T[i] = standardise_values(x.T[i])	
z = np.ones((1338,1))			#create intercept column initialised to 1
X = np.append(z,x,axis = 1)		#append the column at the start of the feature matrix
Y = standardise_values(y)		#standardise target variable
#--------------#

start = time.time()
print("\nCalculating results from solving by normal equations:")
normalarray = np.empty(20)
for i in range(20):
	X_train, X_test, y_train, y_test = my_train_test_split(X, Y, 0.70)
	
	normalarray[i] = normal_equation_solver(X_train, y_train, X_test, y_test)


print("Mean RMSE = ",np.mean(normalarray))
print("Variance of RMSE = ",np.var(normalarray))
print("Time taken to build 20 regression models = ",round(time.time() - start, 2),"s")
print("Done\n")
#-----------#

start = time.time()
print("\nCalculating results from solving by gradient descent:")
gradarray = np.zeros(20)
for i in range(20):
	X_train, X_test, y_train, y_test = my_train_test_split(X, Y, 0.70)
	
	gradarray[i] = gradient_descent(X_train, y_train, X_test, y_test)


print("Mean RMSE = ",np.mean(gradarray))
print("Variance of RMSE = ",np.var(gradarray))
print("Time taken to build 20 regression models = ",round(time.time() - start, 2),"s")
print("Done\n")
#-----------#

start = time.time()
print("\nCalculating results from solving by stochastic gradient descent:")
stocgradarray = np.zeros(20)
for i in range(20):
	X_train, X_test, y_train, y_test = my_train_test_split(X, Y, 0.70)
	
	stocgradarray[i] = stochastic_gradient_descent(X_train, y_train, X_test, y_test)

print("Mean RMSE = ",np.mean(stocgradarray))
print("Variance of RMSE = ",np.var(stocgradarray))
print("Time taken to build 20 regression models = ",round(time.time() - start, 2),"s")
print("Done\n")
#-----------#