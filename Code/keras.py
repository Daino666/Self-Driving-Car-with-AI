# -*- coding: utf-8 -*-
"""Keras.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bEaObWoepRYDvBDZUacRevkDMJQEkFOH
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# %matplotlib inline

no_pts = 600
np.random.seed(0)

Xa = np.array([np.random.normal(13, 2, no_pts),
               np.random.normal(12, 2, no_pts)]).T
Xb = np.array([np.random.normal(6, 2, no_pts),
               np.random.normal(5, 2, no_pts)]).T
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(no_pts), np.ones(no_pts))).T

plt.scatter(Xa[:, 0], Xa[:, 1])
plt.scatter(Xb[:, 0], Xb[:, 1])

model = Sequential()
model.add(
    Dense(units = 1, input_shape= (2,), activation = "sigmoid" )
    )

adam = Adam()
model.compile(adam, loss = "binary_crossentropy", metrics = ["accuracy"])
h = model.fit(x=X, y=Y, verbose = 1, batch_size = 50, epochs = 500, shuffle = "true")

plt.plot(h.history["accuracy"])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["accuracy"])

plt.plot(h.history["loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["loss"])

def plot_decision_boundires(X, model):
  x_span = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1)
  Y_span = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1)

  xx, yy = np.meshgrid(x_span, Y_span)
  xx_ , yy_ = np.ravel(xx), np.ravel(yy)

  Grid = np.c_[xx_, yy_] # use [] not ()

  predicitons = model.predict(Grid)
  '''
    print(predicitons)
   predicions have the shape of

   [
    [a,b]
    [a,c]
    [a.d]
   ]

   so we want to make it a shape of 50*50
  '''

  z = predicitons.reshape(xx.shape)

  plt.contourf(xx ,yy, z)

plot_decision_boundires(X, model)

plot_decision_boundires(X, model)
plt.scatter(Xa[:, 0], Xa[:, 1])
plt.scatter(Xb[:, 0], Xb[:, 1])

point =  np.array([[2.5, 3.5]])
a = model.predict(point)

plt.plot(2.5, 2.5, marker = 'o', markersize = 10 , color ='red' )

print("prediction is ", a )

