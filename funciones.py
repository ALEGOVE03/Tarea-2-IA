'''
Se agrega una clase con las funciones de activación y sus respectivas
derivadas, las funciones utilizadas son:

- Sigmoid
- Tanh
- Relu
'''

import sys  # Para cerrar el programa si hay un error
import numpy as np

# Se declara la función sigmoide
sigmoid = (lambda x: 1 / (1 + np.exp(-x)),
           lambda x: sigmoid[0](x) * (1 - sigmoid[0](x)))

tanh = (lambda x: np.tanh(x),
        lambda x: 1 - (np.tanh(x)) ** 2)

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

ReLU = (lambda x: np.maximum(0, x),
        lambda x: der_ReLU(x))

def der_linear(x):
    temp = []
    for i in range(len(x)):
        temp.append([1])
    return np.array(temp)

linear = (lambda x: x, lambda x: der_linear(x))

def function(nombre):
    if nombre == 'sigmoid':
        return sigmoid
    elif nombre == 'tanh':
        return tanh
    elif nombre == 'ReLU':
        return ReLU
    elif nombre == 'linear':
        return linear
    else:
        print("Incorrect activation function parameter")
        sys.exit(1)


'''
Funciónes de pérdida y sus derivadas, las utilizadas son:

- Regularización L2
'''

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: 2 * (Yp - Yr))
