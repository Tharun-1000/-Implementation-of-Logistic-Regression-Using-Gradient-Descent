# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
![logistic regression using gradient descent](sam.png)
![273632361-914d94f2-effb-4172-bf91-d3881be0129c](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/bd663eba-7715-42a7-85d6-20f350112e4f)
![273632465-4c3a74a6-05e4-4c29-82bf-b3f5fd515c01](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/9fbd6104-d769-430c-9f22-144d0b8724d4)
![273632514-0d784217-b371-43c9-8359-05956190c002](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/c648ea68-5106-44f7-ad6b-0e746fa0dd18)
![273632554-dcf2473b-962f-44fb-bf1c-4541230186de](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/bd7bba72-4653-4967-a021-71aec7bb48e7)
![273632592-a72cadd7-50ca-4355-8dcd-85d74038fa49](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/f31caa56-38b7-4131-b52e-81b0e1886a4f)
![273632662-86158786-18da-4d81-8b8e-12f859b7a116](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/796c26ed-b85a-4b17-8f82-3174bd825f6c)
![273632695-00f12e94-2fd7-49e2-bfb7-d121521f4c3e](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/ccb63ba1-3857-4657-a74b-cc402e9ae751)
![273632747-c5b2fe28-701b-44fc-a953-f1fa5e6723c5](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/b1e2a331-88d4-446a-abee-a916ece6dbbb)
![273632786-dd605d25-c387-4606-990f-88968dba59b7](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/d2711e23-1d1d-4802-b34c-309dbcfe57b8)
![273632822-c78afc2e-f098-4e59-b95a-8dd0a776a3bc](https://github.com/Tharun-1000/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/135952958/1c3784e6-c2fb-4862-9297-da0c6942428d)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

