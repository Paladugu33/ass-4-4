import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=pd.read_csv("candy-data.csv")
x=np.asmatrix(x)
x=np.array(x)
N,D=np.shape(x)
X=np.zeros((N,D-2))
X[:,0:10]=x[:,1:11]
Y=x[:,11]

for i in range (len(X[:,9])):
    if X[i,9]<0.45:
        X[i,10]=0
    elif 0.45<=X[i,9]<0.86 :
        X[i,10]=1
    elif X[i,9] >=0.86:
        X[i,10]=-2
X=np.array(X,dtype=float)
Y=np.array(Y,dtype=float)
X[:,10]=X[:,10]*100
w=np.linalg.solve(np.dot(X[:,0:10].T,X[:,0:10]),np.dot(X[:,0:10].T,Y))
X[:,[9,10]] = X[:,[10,9]]
Y1=np.dot(X[:,:10],w)
print(w)
def tanh(a):
    return (1-np.exp(-2*a))/(1+np.exp(-2*a))
def softmax(a):
    expa=np.exp(a)
    return expa/expa.sum()
def sigmod(a):
    return (1/(1+np.exp(-a)))
Ysi= sigmod(Y1)
Yso = softmax(Y1)
Ytn = tanh(Y1)
print ("tanhh values are: ",Ytn)
print ("softmax values are: ",Yso)
print ("Sigmoid values are: ",np.round(Ysi))
plt.scatter(np.sort(x[:,0].T),np.sort(Ysi.T))
plt.plot(np.sort(x[:,0].T),np.sort(Ysi.T))
plt.show()