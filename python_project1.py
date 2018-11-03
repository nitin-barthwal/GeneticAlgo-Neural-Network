# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:12:29 2018
"""
# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#1 – Import the dataset from the .csv file provide .
dataframes = pd.read_csv('Project 1 - Dataset.csv', header='infer' )

#2 Choose N=10 (see STRUCTURE on slide #2) .    
N=10   

#3 – Only select the first 5 columns as input, and the very last column as output (target). You can eliminate the rest of the columns.
s= dataframes.iloc[:,0:5]
y=dataframes.iloc[:,13]
P=5

#4 - Choose 25% of the dataset (random) as testing and the rest 75% as training samples. Leave the testing dataset on the side for the time being

X_train, X_test, Y_train, Y_test = train_test_split(s, y, test_size=0.25 )

weightLbs = X_train[['Weight lbs']]
heightInch = X_train[['Height inch']]

#4 – Normalise the training dataset with values between 0 and 1.

xMaxValue, xMinValue = X_train.max(), X_train.min()
X_train = (X_train - xMinValue) / (xMaxValue - xMinValue)

yMaxValue, yminValue = Y_train.max(), Y_train.min()
Y_train = (Y_train - yminValue) / (yMaxValue - yminValue)

X_train= np.matrix(X_train)
# Reset Index
Y_train = Y_train.reset_index(drop=True) 

xMaxValue, xMinValue = X_test.max(), X_test.min()
X_test = (X_test - xMinValue)/(xMaxValue - xMinValue)

yMaxValue, yminValue = Y_test.max(), Y_test.min()
Y_test = (Y_test - yminValue)/(yMaxValue - yminValue)

X_test_initial = X_test
Y_test_initial = Y_test

X_test= np.matrix(X_test)

# Reset Index
Y_test = Y_test.reset_index(drop=True)

#5 – Calculate the number of parameters (weights) you need to tune in the STRUCTURE (refer to slide #2).
weights=P*N

#6 – Calculate Equation 1
def calculate_eq1Function(temp):
    var2 = 0
    var3 = 0
    for i in range(N):
        x = temp.item(i)
        var3 =  1 + np.exp(-x) 
        var2 = var2 + (1 / var3 ) 
    return var2

# Calculate fitness value for given weight
def FitnessValueCalculation(w):
 yhat_sum = 0
 yhat_sq = 0
 d = 0
 for j in range(train):
         temp = np.matmul (  X_train[[j],:]  ,  w)
         yhat_sum = calculate_eq1Function(temp)
         d = d + j
         yhat_sq = yhat_sq + np.square( yhat_sum - Y_train[j])
 wmax,wmin = w.max(),w.min()
 normalized_w = w
 normalized_w = (normalized_w - wmin)/(wmax-wmin)
 # convert to np matrix
 normalized_w= np.matrix(normalized_w)
 normalized_w = np.around(normalized_w*1000)
 # Reshape
 c = normalized_w.reshape(1,-1)
 # step 11
 chromosome = convertFunction(c)
 
 fitness_value = (1- (yhat_sq/train))*100
 return chromosome,fitness_value,d


# Test length given in test dataset
test = len(X_test)
yhat_sumlist=list()

def convertFunction(val):
    chromosome=""
    for j in range(weights):
         chr = bin(int(val.item(j)))[2:].zfill(10)
         chromosome=chromosome+str(chr)+''
    return chromosome

def Overall_error_calculation(wght):
 yhat_sq = 0
 yhat_sum = 0
 
 yhat_sumlist.clear()
 for i in range(test):
         temp = np.matmul (X_test[[i],:]  ,  wght)
         yhat_sum = calculate_eq1Function(temp)
         yhat_sumlist.append( yhat_sum)
         yhat_sq = yhat_sq + np.square( yhat_sum - Y_test[i])
 overall_error_test = (yhat_sq/test)
 return overall_error_test

global_test_fitness = list()
test_chromosome_list1 = []

#Crossover function
def crossoverFunction(par1,par2):
    crossover_point = int(len(par1)/2)
    partial1= par1[0:crossover_point] + par2[crossover_point:]
    partial2 = par2[0:crossover_point] + par1[crossover_point:]
    return partial1, partial2

# Mutating
def mutateFunction(list1):
     l1=list(list1)
     listing =   random.sample(range(0, len(l1)), int(len(l1) * 0.05))
     for i in listing:
          if l1[i]=='0':
              l1[i]='1'
          else:
              l1[i]='0'
     list3 = ''.join(l1)
     return list3;

# Debinarizing
def debinarizationFunction(var1):
    Arr1 = []
    lower= -1
    upper = 1
    for i in range(0,weights*10,10):
        value1 = int(var1[i:i+10],2)/1000
        Arr1= np.append(Arr1,value1)
    normalized_value = [lower + (upper - lower) * value1 for value1 in Arr1]  
    return  normalized_value
           
Npop =500
train = 189
fittest = -1
learnerWeight=list()
Z_normalized=list()
chromosome_list=list()
chromosome_list1 = list()
chromosome_list2 = list()
fitness_value=list()
fitness_value2=list()
global_fitness = list()
learner_weight = list()

for u in range(Npop):
    learnerWeight.append(np.matrix((np.random.uniform(low=-1,high=1,size=(5,10)))))

for index in range(Npop):
    chromosome, f, d= FitnessValueCalculation(np.array(learnerWeight[index]))
    fitness_value.append(f)
    if fittest == -1 :
            fittest = f 
            parent = learnerWeight[index]
            parent_chromosome = chromosome 
            final_best_weight = learnerWeight[index]
    elif fittest < f:
            fittest = f
            global_fitness.append(fittest)
            parent = learnerWeight[index]
            parent_chromosome = chromosome
            final_best_weight = learnerWeight[index] 
    chromosome_list1.append(chromosome)        
    

for index in range(Npop): 
    current_cr = chromosome_list1[index] 
    a,b = crossoverFunction(parent_chromosome,current_cr)
    c = mutateFunction(a)
    d = mutateFunction(b)
    # step 14 debinarization

    X_normalized = debinarizationFunction(c) 
    Y_normalized = debinarizationFunction(d) 
    Y_normalized = np.matrix(Y_normalized)
    X_normalized = np.matrix(X_normalized)
    Z_normalized.append(X_normalized.reshape(5,10))
    Z_normalized.append(Y_normalized.reshape(5,10))


# Running the loop 15 times only
l =15
print("Total Loop :: ",l)
for index in range(l):
    print(" Loop no ",index)
    fitness_value.clear()
    chromosome_list1.clear()
    chromosome_list.clear()
    chromosome_list2.clear()
    fitness_value2.clear()
    learner_weight.clear()
    
    for u in range(len(Z_normalized)):
        chromosome, f, d = FitnessValueCalculation(np.array(Z_normalized[u]))
        fitness_value.append(f)
        if fittest == -1 :
            parent = Z_normalized[u]
            parent_chromosome = chromosome 
            fittest = f
            final_best_weight = Z_normalized[u]
        elif fittest < f:
            global_fitness.append(fittest)
            parent = Z_normalized[u]
            parent_chromosome = chromosome
            fittest = f 
            final_best_weight = Z_normalized[u]
         
        chromosome_list1.append(chromosome) 
        learner_weight.append(Z_normalized[u])
    Z_normalized.clear()
    print('Fittest Value for iteration - ',index , 'is' , fittest)    
    fitness_value2 = fitness_value   
    fitness_value2.sort()
    mid = fitness_value2[int(len(fitness_value2)/2)]
    len(fitness_value2)
    chromosome_list2.clear();
    for x in range(len(fitness_value)):
        if len(chromosome_list2) >=len(fitness_value2)/2:
            break;
        if float(mid) <= float(fitness_value[x]):    
                chromosome_list2.append(chromosome_list1[x])   
    for u in range(len(chromosome_list2)):
        current_cr = chromosome_list2[u]
        a,b = crossoverFunction(parent_chromosome,current_cr)
        c = mutateFunction(a)
        d = mutateFunction(b)
    # Step 14 Do the de-binarization of the chromosomes according to following procedure :
    #I ) De-segment each chromosome to its 10-bits components.
    #II ) Make a binary to decimal conversion of each single 10-bit weight.
    #III ) Divide them by 1000
    #IV ) De-normalise weights to values between -1 and 1
        X_normalized = debinarizationFunction(c) 
        Y_normalized = debinarizationFunction(d)
        Y_normalized = np.matrix(Y_normalized)
        X_normalized = np.matrix(X_normalized)
        Z_normalized.append(X_normalized.reshape(5,10))
        Z_normalized.append(Y_normalized.reshape(5,10))
        
 # Plot for fitness values

plt.scatter( global_fitness,range(len(global_fitness)), marker='^',color='red' )


plt.title('2D Scatter plot Project 1 ')
plt.ylabel('No of Iterations')
plt.xlabel('Fitness Value')
plt.show()       

overall_error = Overall_error_calculation(final_best_weight)

# Plot 3d Scatter Plot
fig = plt.figure()
ax = Axes3D(fig)
weightLbs = X_test_initial[['Weight lbs']]
heightInch = X_test_initial[['Height inch']]
dataframes_f = pd.DataFrame({'fitness':yhat_sumlist})
ax.set_xlabel('Weight Values')
ax.set_ylabel('Height Values')
ax.set_zlabel('Y Output Values')
ax.scatter(weightLbs, heightInch, Y_test, marker='^',color='red')
ax.scatter(weightLbs, heightInch, dataframes_f, marker='^',color='green')
plt.show()

# Showing overall error
print( "Overall Error " ,overall_error)