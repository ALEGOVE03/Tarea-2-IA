import numpy as np
import matplotlib.pyplot as plt
import neurona as nn

data = np.genfromtxt('Datos/airfoil.dat')

print(data)

red = [nn.neural_layer(len(data[0][:-1]), 4, 'sigmoid'),
       nn.neural_layer(4, 8, 'sigmoid'),
       nn.neural_layer(8, 1, 'linear')]

loss, valid = nn.train(red, data, 500, 10, 0.2, 0.00001)

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
