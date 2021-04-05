import numpy as np
import matplotlib.pyplot as plt
import neurona as nn

data = np.genfromtxt('Datos/erosion.data')

print(data)
print()

red = [nn.neural_layer(len(data[:-1]), 4, 'sigmoid'),
       nn.neural_layer(4, 1, 'linear')]
