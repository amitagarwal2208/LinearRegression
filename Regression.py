#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ### Data(X) -> Acidity of milk
# ### Data(Y) -> Density of milk

# In[13]:


def readData(filename):
    # df stands for data frame
    df = pd.read_csv(filename)
    # Treat df as a table
    return df.values     ## Converts it into a numpy array and returns that array.

x = readData("/Volumes/part3/Data_Milk_Regression1/linearX.csv")
print(x.shape)
x = x.reshape((99,))
print(x.shape)
print()
print(x)
print() 
print()
y = readData("/Volumes/part3/Data_Milk_Regression1/linearY.csv")
y = y.reshape((99,))
print(y)


# #Plotting the Data

# 
# 

# In[14]:


plt.plot(x,y)


# ##How does plot work?
# 

# In[17]:


a=[1,2,3]
b=[3,6,9]

plt.scatter(a,b)


# In[18]:


plt.plot(a,b)   # 1 is mapped to 3 , 2 with 6 and so on


# #Coming back to our code
# 

# In[19]:


plt.plot(x,y)


# In[20]:


plt.scatter(x,y)


# In[33]:


##Normalisation --> Make mean=0 and standard deviation = 1
##It is done so that data doesn;t get crowded at one place and becomes scattered so that
##we can see it clearly

## It also brings data near origin --> convergance will come fast
#print(x.std())
x = x-x.mean()/(x.std())

plt.scatter(x,y)  # Check that data is near origin
plt.show()


# In[34]:


print(x.shape)


# In[73]:


## Linear Regression Algorithm

X = x
Y = y

def hypothesis(theta , x):   
    # x is a point.
    # theta is an array
    # This function calculates the hypothesis function
    
    return theta[0] + theta[1]*x   # Since it has only 1 attribute
    
def error(X,Y,theta):
    #X and Y are training sets/data
    
    # Total error is (sigma from 1 to m)(h(x)-y)
    
    total_error = 0 ;
    m = X.shape[0]   # Where m is the number of examples
    
    for i in range(m):
        total_error+=(Y[i] - hypothesis(theta , X[i]))**2
        
    return 0.5*total_error

def CalcGradient(X,Y,theta):
    
    grad = np.array([0.0,0.0])
    m=X.shape[0]
    for i in range(m):
        grad[0]+=hypothesis(theta , X[i]) - Y[i]
        grad[1]+=( hypothesis(theta , X[i]) - Y[i])*X[i]      
    
    return grad


def gradientDescent(X,Y,learningRate , maxItr):
    grad = np.array([0.0,0.0])
    # It stores the 2 gradients dJ(theta)/dtheta0 and dJ(theta)/dtheta1
    
    theta = np.array([0.0,0.0])
    # It stores theta0 and theta 1
    e=[]
    for i in range(maxItr):
        grad = CalcGradient(X,Y,theta)
        
        # theta = theta - (learningRate)*(gradient)
                   
            
        ce = error(X,Y,theta)    
        theta[0]-=(learningRate)*grad[0]
        theta[1]-=(learningRate)*grad[1]
        e.append(ce)
        
        
        
    return theta,e
        
theta,e = gradientDescent(X,Y,0.001,80)
print(theta[0] , theta[1])
                   
                   


# In[71]:


plt.scatter(X,Y)
plt.plot(X,hypothesis(theta , X),color='r')
plt.show()


# In[72]:


plt.plot(e)


# In[ ]:




