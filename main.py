import numpy as np
import matplotlib.pyplot as plt
import neurona as nn

data = np.genfromtxt('Datos/erosion.data')

print(data)
print()

X, Y, X_valid, Y_valid = nn.split_data(data, 0.3)
