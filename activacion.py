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
    resul = []
    for i in x:
        if i > 0:
            resul.append(1)
        else:
            resul.append(0)
    return resul

ReLU = (lambda x: np.maximum(0, x),
        lambda x: der_ReLU(x))

linear = (lambda x, b: x + b,
         1, 1)

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
