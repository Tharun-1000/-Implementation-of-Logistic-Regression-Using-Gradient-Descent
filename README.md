# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Use the standard libraries in python for finding linear regression


2. Set variables for assigning dataset values.


3. Import linear regression from sklearn.

4. Predict the values of array.


5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.


6. Obtain the graph
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: THARUN K
RegisterNumber: 212222040172
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

x[:5]

y[:5]

plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="cadetblue")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted",color="plum")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot),color="cadetblue")
plt.show()

def costFunction(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,x,y):

  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad


x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted",color="mediumpurple")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted",color="pink")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,x,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return(prob>=0.5).astype(int)

np.mean(predict(res.x,x)==y)

*/
```

## Output:

1.Array value of x

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/b63db0b9-b3f4-45d3-9c23-05dcdcb5fb1d)

2.Array Value of y

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/20c2a6c2-2894-495e-ae10-7d37c1aa88d8)

3.Exam 1-Score Graph

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/852ff766-78eb-4baf-ba95-11b9eed7847b)

4.Sigmoid function graph

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/0355fcd1-7469-4236-895e-554290672679)

5.x_train_grad value

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/f73e2700-53d6-44b0-80cc-5fd02b591a7b)

6.y_train_grad value

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/0cbcb04b-874f-4659-81e0-f91bf7ca6d1e)

7.Print res.x

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/990798d5-f2c0-451f-b443-8cae84eb9782)

8.Decision boundary-graph for exam score

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/61a11edd-de96-43e4-9a39-67aa6f111f66)

9.Probability value

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/5b527617-ac54-43b3-8c57-6ab45902c774)

10.Prediction value of mean

![image](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/9c68b61c-71af-4f3d-9131-653a7f85c7da)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

