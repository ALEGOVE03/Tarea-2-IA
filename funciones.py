'''
Se agrega una clase con las funciones de activación y sus respectivas
derivadas, las funciones utilizadas son:

- Sigmoid
- Tanh
- Relu
'''

import sys  # Para cerrar el programa si hay un error
import numpy as np # Trabajar con matrices

# Se declara la función sigmoide y su derivada
sigmoid = (lambda x: np.around(1 / (1 + np.exp(-x, dtype=np.float64))),
           lambda x: sigmoid[0](x) * (1 - sigmoid[0](x)))

# Se declara la función tanh y su derivada
tanh = (lambda x: np.tanh(x),
        lambda x: 1 - (np.tanh(x)) ** 2)

# Se define la derivada de la función ReLU de forma matricial
def der_ReLU(x):
    temp = len(x[0])
    resul2 = []
    for i in range(len(x)):
        resul = []
        for j in range(temp):
            if x[i][j] > 0:
                resul.append(1)
            else:
                resul.append(0)
        resul2.append(resul)
    return resul2

# Se define la función ReLU con si derivada
ReLU = (lambda x: np.maximum(0, x),
        lambda x: der_ReLU(x))

# Función para devolver la función deseada a trabajar en la capa de la red
def function(nombre):
    if nombre == 'sigmoid':
        return sigmoid
    elif nombre == 'tanh':
        return tanh
    elif nombre == 'ReLU':
        return ReLU
    else:
        print("Incorrect activation function parameter")
        sys.exit(1)


'''
Funciónes de pérdida y sus derivadas, las utilizadas son:

- Regularización L2
'''

# Se define la función de pérdida L2
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: 2 * (Yp - Yr))
