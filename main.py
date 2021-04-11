import numpy as np
import matplotlib.pyplot as plt
import neurona as nn
import random


data = np.genfromtxt('Datos/airfoil.dat')

'''
# Para le caso de los O-rings se debe acomodar los datos
cop_dat = []

for i in range(len(data)):
    temp = []
    for j in range(len(data[i])):
        if j != 1:
            temp.append(data[i][j])
    temp.append(data[i][1])
    cop_dat.append(temp)

data = np.array(cop_dat)
cop_dat = []
'''

red = [nn.neural_layer(len(data[0][:-1]), 4, 'sigmoid'),
       nn.neural_layer(4, 8, 'sigmoid'),
       nn.neural_layer(8, 1, 'linear')]

loss, valid, R2 = nn.train(red, data, 100, 10, 0.3, 0.00001, True)

axis = []
axis2 = []

for i in range(len(loss)):
    if i % 10 == 0:
        axis2.append(i)
    axis.append(i)

z = nn.predict(red, data[0][:-1])
print(z)
print(loss[-1])
plt.plot(axis, loss)
plt.plot(axis2, valid)
plt.show()

# Guardar los pesos de la red

nn.save(red, "Pesos/NN.txt")

red2 = nn.load("Pesos/NN.txt")
z = nn.predict(red2, data[0][:-1])
print(z, "\n")
print(R2)
