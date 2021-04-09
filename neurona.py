import funciones as ft  # Se importan las funciones de activación
import numpy as np
import random

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: 2 * (Yp - Yr))

'''
Clase para crear una capa de la red neuronal
Parametros de inicialización:
- n_conn: Número de conexiones que va a tener cada neurona de la capa
-- es decir, si de la capa anterior se tiene 3 neuronas cada neuronal
-- de la capa actual tendría 3 conexiones por neurona

- n_neur: Número de neuronas de la capa

- act_f: Función de activación para cada neurona de la capa
'''

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = ft.function(act_f)
        self.theta = np.random.rand(1, n_neur) * 2 - 1
        self.w = np.random.rand(n_conn, n_neur) * 2 - 1

# Divide los datos entre entrenamiento y validación
def split_data(data, split):
    np.random.seed()
    data = np.random.permutation(data)
    train, validate = np.split(data, [int((1-split)*len(data))])

    # Se acomodan los datos entre validación
    data_train = train[:, :-1]
    copia = []
    for i in range(len(train)):
        copia.append([train[i, -1]])

    y_train = np.array(copia)

    data_valid = validate[:, :-1]
    copia = []
    for i in range(len(validate)):
        copia.append([validate[i, -1]])

    y_valid = np.array(copia)
    # Final de reacomodar datos
    return data_train, y_train, data_valid, y_valid

'''
Entrenamiento de la red neuronal
Entradas
- neural_net: Red neuronal a entrenar
- data: Datos para el entrenamiento
- num_iter: Número de iteraciones
- num_validation: Cada cuanto se hace una validación
- val_size: Porcentaje de los datos utilizados para validación
- l2_cost
'''

def train(neural_net, data, num_iter, num_validation, val_size, lr=0.001):

    X, Y, X_valid, Y_valid = split_data(data, val_size)
    loss_train = []
    loss_valid = []

    for i in range(num_iter):
        out = [(None, X)]

        # Forwad pass
        for l, layer in enumerate(neural_net):
            z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
            a = neural_net[l].act_f[0](z)
            out.append((z, a))

        # Se encarga de actualizar los pesos
        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):
            a = out[l+1][1]
            if (l == len(neural_net) - 1):
                # Calcular delta última capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].w

            # Gradient descent
            neural_net[l].theta = neural_net[l].theta - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - lr * out[l][1].T @ deltas[0]

        # Guardar datos de pérdida
        loss_train.append(np.mean(l2_cost[0](out[-1][1], Y)))

        # Hacer pérdida de validación
        if i % num_validation == 0 and num_validation != 0:
            out = [(None, X_valid)]

            # Forwad pass
            for l, layer in enumerate(neural_net):
                z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
                a = neural_net[l].act_f[0](z)
                out.append((z, a))

            # Guardar pérdida de validacion
            loss_valid.append(np.mean(l2_cost[0](a, Y_valid)))

    return [loss_train, loss_valid]

# Hacer prediciones de la red neuronal, ya entrenada
def predict(neural_net, X):
    out = [(None, X)]
    # Forwad pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    return np.mean(a)
