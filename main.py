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
       nn.neural_layer(4, 2, 'sigmoid'),
       nn.neural_layer(2, 1, 'linear')]

num_iter = 50
num_valid = num_iter // 10

loss, valid, R2 = nn.train(red, data, num_iter, num_valid, 0.3, 0.00001, False)

x_loss = []
x_val = []

for i in range(len(loss)):
    if i % num_valid == 0:
        x_val.append(i)
    x_loss.append(i)

plt.plot(x_loss, loss)
plt.plot(x_val, valid)
plt.show()

# ================ Guardar curvas de périda ================

file = open("Curvas/LossTrain.txt", "w")
for i in range(len(loss)):
    file.write(str(x_loss[i]) + " " + str(loss[i]) + "\n")
file.close()

file = open("Curvas/LossValid.txt", "w")
for i in range(len(valid)):
    file.write(str(x_val[i]) + " " + str(valid[i]) + "\n")

file.close()

# ================ Guardar valor R2  ================
file = open("Curvas/R2.txt", "w")
file.write(str(R2))
file.close()

# Guardar los pesos de la red
print(loss[-1])
print(R2)

nn.save(red, "Pesos/NN.txt")

red2 = nn.load("Pesos/NN.txt")
