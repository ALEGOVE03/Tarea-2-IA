import funciones as ft  # Se importan las funciones de activación
import numpy as np # Libreria para trabajar con matrices
import random # Para generar valores aleatorios
import os # Para crear directorios en el caso de ser necesario
import errno # Para trabajar con errores

# Se carga la función de pérdida L2 del archivo funciones
l2_cost = ft.l2_cost

'''
Clase para crear una capa de la red neuronal
Parametros de inicialización:
- n_conn: Número de conexiones que va a tener cada neurona de la capa
-- es decir, si de la capa anterior se tiene 3 neuronas cada neuronal
-- de la capa actual tendría 3 conexiones por neurona

- n_neur: Número de neuronas de la capa

- act_f: Función de activación para cada neurona de la capa

- funct: Es un string que permite guardar la red neuronal sin perder
la función de activación de la capa
'''

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = ft.function(act_f)
        self.theta = np.random.rand(1, n_neur)
        self.w = np.random.rand(n_conn, n_neur)
        self.funct = act_f

# Divide los datos entre entrenamiento y validación
def split_data(data, split):
    np.random.seed() # Crea una semilla para el random
    data = np.random.permutation(data) # Mezcla los datos del conjunto
    # Divide los datos entre train y validación
    train, validate = np.split(data, [int((1-split)*len(data))])

    # Se acomodan los datos de entrada en el entrenamiento
    data_train = train[:, :-1]
    copia = []
    # Se obtienen las salidas para cada valor de entrada
    for i in range(len(train)):
        copia.append([train[i, -1]])

    # Se define las salidas para los datos de entrenamiento
    y_train = np.array(copia)

    # Se acomodan los datos de entrada para la validación
    data_valid = validate[:, :-1]
    copia = []
    # Se obtienen las salidas para cada valor de entrada
    for i in range(len(validate)):
        copia.append([validate[i, -1]])

    # Se definen las salidas para los datos de entrenamiento
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
- lr: Tasa de aprendizaje
- gauss: Variable booleana para definir si es trabaja con una tasa variable
'''

def train(neural_net, data, num_iter, num_validation, val_size, lr=0.001, gauss=False):

    # Se dividen los datos para entrenamento y validación
    X, Y, X_valid, Y_valid = split_data(data, val_size)

    # Arreglos para guardar la pérdida de entrenamiento y de validación
    loss_train = []
    loss_valid = []

    pesos = [[], [], [], []]

    # Ciclos que controla la cantidad de entrenamientos para la red
    for i in range(num_iter):

        # En el caso que se desee trabajar con una tasa variable gaussiana
        if gauss:
            lr = random.gauss(lr, lr * 0.25)

        # Arreglo en el que se guardan los valores de net (z) y la salida
        # de cada capa (a), al inicio no se tiene un (z) y la salida se
        # se define como la entrada a la red
        out = [(None, X)]

        # Forwad pass de la red
        for l, layer in enumerate(neural_net):
            z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
            a = neural_net[l].act_f[0](z)
            out.append((z, a))


        deltas = []
        # Backpropagation y desceso por gradiente
        for l in reversed(range(0, len(neural_net))):
            # Se obtiene la salida de cada capa, se obtuvo en el fordware
            a = out[l+1][1]
            # Condicional para el caso en que se trabaja con la capa de salida
            if (l == len(neural_net) - 1):
                # Calcular delta última capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            # Guarda los pesos de la capa anterior
            # esto se debe a que los pesos se actualizan al mismo tiempo
            # del backpropagation
            _W = neural_net[l].w

            # Se obtienen los valores de los pesos para el estudio de la evolución
            if l == 1:
                if i <= 10:
                    temp = - lr * out[l][1].T @ deltas[0]
                    # Actualización de los pesos
                    pesos[0].append(temp[0][0])
                    # El peso actual
                    pesos[1].append(neural_net[1].w[0][0])
                    # La entrada (net) a el perceptron
                    pesos[2].append(out[1][0][0][0])
                    # La salida de la neurona del perceptron
                    pesos[3].append(out[1][1][0][0])

            # Gradient descent
            # Se encarga de actualizar los pesos de la capa de lal red
            # y del sesgo de cada capa
            neural_net[l].theta = neural_net[l].theta - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - lr * out[l][1].T @ deltas[0]

        # Guardar datos de pérdida
        loss_train.append(l2_cost[0](out[-1][1], Y))

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

    # Se obtiene el coeficiente de determinación
    out = [(None, X)]

    # Forwad pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    # Se obtiene el error cuadrático del valor final
    loss = l2_cost[0](a, Y)
    # Se obtiene el valor medio de los datos reales de los datos
    y_mean = np.mean(Y)
    y_loss = l2_cost[0](Y, y_mean)

    R2 = 1 - (loss/y_loss) # Obtención del R2

    # Regresa valores relevantes en la red
    return [loss_train, loss_valid, R2, pesos]

# Hacer prediciones de la red neuronal, ya entrenada
# neural_net: Red neuronal
# salida: valor utilizado para normalizar las salidas
# escalamiento: valor con el que se normalizó los datos
def predict(neural_net, X, salida):
    out = [(None, X)]

    # Forwad pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].w + neural_net[l].theta
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    # Se regrea el valor predicho
    return np.mean(a) * salida

# Guardar pesos y bias de la red neuronal
def save(neural_net, direction, escalamiento):
    inicio = False
    direc = ""

    # Se crea el directorio donde se guardan los pesos de la red
    for i in reversed(range(len(direction))):
        if inicio:
            if direction[i] == "/":
                break
            else:
                direc += direction[i]
        if direction[i] == ".":
            inicio = True

    try: # Error en el caso que el directorio ya exista
        os.mkdir('Pesos/' + direc)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Crea archivo donde se guardan valores importantes para el load
    file = open(direction, "w")
    file.write(str(len(neural_net)))

    # Se guarda cada peso y sesgo por capa de la red neuronal
    for i in range(len(neural_net)):
        dir2 = "Pesos/" + direc + "/" + direc + "_W" + str(i) + ".txt"
        np.savetxt(dir2, neural_net[i].w)
        dir2 = "Pesos/" + direc + "/" + direc + "_b" + str(i) + ".txt"
        np.savetxt(dir2, neural_net[i].theta)
        file.write("\n" + neural_net[i].funct)

    file.write("\n" + str(escalamiento))
    file.close()

# Cargar el modelo, pesos y sesgos de una red neuronal
def load(direction):
    inicio = False
    direc = ""

    # Se crea el directorio donde se guardan los pesos de la red
    for i in reversed(range(len(direction))):
        if inicio:
            if direction[i] == "/":
                break
            else:
                direc += direction[i]
        if direction[i] == ".":
            inicio = True

    datos = []
    # Leer datos del archivo raíz
    # En este archivo se encuentran la cantidad de capaz con su
    # función de activación respectiva
    with open(direction) as file:
    	lineas = file.readlines()
    	for linea in lineas:
    		datos.append(linea.strip('\n'))

    # El valor utilizado para la normalización de los datos
    escalamiento = datos[-1]
    red = []

    # Carga cada peso y sesgo de las capas de la red
    for i in range(int(datos[0])):
        dir = "Pesos/" + direc + "/" + direc + "_W" + str(i) + ".txt"
        matriz = np.loadtxt(dir)
        red.append(neural_layer(1, 1, datos[i + 1]))
        red[-1].w = matriz
        dir = "Pesos/" + direc + "/" + direc + "_b" + str(i) + ".txt"
        matriz = np.loadtxt(dir)
        red[-1].theta = matriz

    # Retorna la red y el valor de la normalización utilizada
    return red, float(escalamiento)
