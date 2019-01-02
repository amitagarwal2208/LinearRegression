import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def readData(filename):
    # df stands for data frame
    df = pd.read_csv(filename)
    # Treat df as a table
    return df.values     ## Converts it into a numpy array and returns that array.

x = readData("/Volumes/part3/Data_Milk_Regression1/linearX.csv")
print(x.shape)
x = x.reshape((99,))
print(x.shape)

y = readData("/Volumes/part3/Data_Milk_Regression1/linearY.csv")
print(y.shape)
y = y.reshape((99,))
print(y.shape)
X= (x-x.mean())/x.std()
Y=y





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
    
    theta = np.array([-3.0,-1.0])
    # It stores theta0 and theta 1
    e=[]
    theta_list=[]
    for i in range(maxItr):
        grad = CalcGradient(X,Y,theta)
        
        # theta = theta - (learningRate)*(gradient)
                   
            
        ce = error(X,Y,theta)    
        theta[0]-=(learningRate)*grad[0]
        theta[1]-=(learningRate)*grad[1]
        e.append(ce)
        theta_list.append((theta[0],theta[1]))
        
        
    return theta,e,theta_list
        
theta,e,theta_list = gradientDescent(X,Y,0.001,500)

plt.scatter(X,Y,label="Training data")
plt.plot(X,hypothesis(theta , X),color='r',label="Prediction")
plt.legend()
plt.show()
print(X.shape , hypothesis(theta , X).shape)





t0 = np.arange(-2,3,0.1)
t1 = np.arange(-2,3,0.1)

t0,t1 = np.meshgrid(t0,t1)
print(t0.shape,t1.shape)
J = np.zeros(t0.shape)
print(J.shape)
m = t0.shape[0]
n = t0.shape[1]

for i in range(m):
    for j in range(n):
        
        J[i,j] = np.sum((Y-t0[i,j]-X*t1[i,j])**2)
#print(type(X),type(Y),type(J))        
print(J.shape)


theta_list = np.array(theta_list)

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.scatter(theta_list[:,0] , theta_list[:,1] , e , c='black' , label="trajectory")
axes.plot_surface(t0,t1,J,cmap="rainbow",alpha=0.5)
plt.legend()
plt.show()

fig = plt.figure()
axes = fig.gca(projection='3d')
#axes.set_xlim([-2,2])
#axes.set_ylim([-2,2])
axes.scatter(theta_list[:,0] , theta_list[:,1] , e , c="black" , label="trajectory")
axes.contour(t0,t1,J)
plt.title("Contours in 3d")
plt.legend()
plt.show()

plt.contour(t0,t1,J)
plt.scatter(theta_list[:,0],theta_list[:,1] , c='k' , marker='^' , label = "trajectory")
plt.legend()
plt.show()

















