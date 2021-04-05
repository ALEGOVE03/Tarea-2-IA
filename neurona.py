import activacion as ft  # Se importan las funciones de activación
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles

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
        if act_f == 'linear':
            self.b = np.random.rand(1, n_neur) * 2 - 1

'''
Mejor no ponerlo
def create_nn(topology, act_f):
    nn = []  # Contiene las capas de la red neurnal
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))
    return nn
'''

# Función de pérdida y su derivada
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: 2 * (Yp - Yr))

def train(neural_net, X, Y, num_validation, num_iter, split, l2_cost, lr=0.001, train=True):

    out = [(None, X)]

    # Forwad pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1]  @ neural_net[l].w + neural_net[l].theta
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    if train:
        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):
            if (l == len(neural_net) - 1):
                # Calcular delta última capa
                a = out[l+1][1]
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
                neural_net.b = neural_net.b - lr * np.mean(l2_cost[1](a, Y) * act_f[2](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].w

            # Gradient descent
            neural_net[l].theta = neural_net[l].theta - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - lr * out[l][1].T @ deltas[0]

def predict(neural_net, X):
    out = [(None, X)]

    # Forwad pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1]  @ neural_net[l].w + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    return out
