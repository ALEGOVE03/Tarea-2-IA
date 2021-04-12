import numpy as np
import matplotlib.pyplot as plt
import neurona as nn
import random


data = np.genfromtxt('Datos/airfoil.dat')
print(data)
data = data.T

for i in range(len(data)):
    x = np.amax(data[i], axis=0)
    data[i] = data[i] / x
    if i == len(data) - 1:
        salida = x

data = data.T

print(data)

red = [nn.neural_layer(len(data[0][:-1]), 4, 'sigmoid'),
       nn.neural_layer(4, 5, 'sigmoid'),
       nn.neural_layer(5, 1, 'linear')]

num_iter = 50
num_valid = num_iter // 10

loss, valid, R2 = nn.train(red, data, num_iter, num_valid, 0.3, 0.00001, False)

axis = []
axis2 = []

for i in range(len(loss)):
    if i % num_valid == 0:
        axis2.append(i)
    axis.append(i)

plt.plot(axis, loss)
plt.plot(axis2, valid)
plt.show()

# Guardar los pesos de la red
print(loss[-1])
print(R2)

nn.save(red, "Pesos/NN.txt")

red2 = nn.load("Pesos/NN.txt")
