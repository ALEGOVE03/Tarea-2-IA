import numpy as np # Trabajar de forma matricial
import matplotlib.pyplot as plt # Imprimir los datos de entrenamiento y de validación
import neurona as nn # Se obtiene el archivo donde está la red programada

# Se obtienen los datos
data = np.genfromtxt('Datos/erosion.data')
# Se muestran los datos para el entrenamiento
print(data)

# Se normalizan los datos de entrada
data = data.T

for i in range(len(data)):
    x = np.amax(data[i], axis=0)
    data[i] = data[i] / x
    if i == len(data) - 1:
        salida = x

# Resultados normalizados
data = data.T

# Se imprimen los valores normalizados
print(data)

# Se define la red con la cantidad de capas a utilizar
red = [nn.neural_layer(len(data[0][:-1]), 4, 'tanh'),
       nn.neural_layer(4, 4, 'tanh'),
       nn.neural_layer(4, 1, 'tanh')]

# Numero de iteraciones de entrenamiento
num_iter = 50
# Número de validaciones
num_valid = num_iter // 10

# Se realiza el entrenamiento de la red y se obtienen las pérdida y
# el coeficiente de determinación
loss, valid, R2, pesos = nn.train(red, data, num_iter, num_valid, 0.3, 0.00001, False)

x_loss = []
x_val = []

# Se crean el eje de las abscisas para el entrenamiento y validación
for i in range(len(loss)):
    if i % num_valid == 0:
        x_val.append(i)
    x_loss.append(i)

# Se grafican los entrenamientos
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

# Imprimir el último valor de pérdida de entrenamiento y el R2
print(loss[-1])
print(R2)

axis = []
for i in range(len(pesos)):
    print(pesos[i], "\n")

print(pesos[1][0] + pesos[0][0])

# ================ Guardar evolución pesos ================
# Iteración / Delta / Peso actual / Entrada perceptron (z)  / Salida perceptron (a)

file = open("Curvas/upgrade.txt", "w")
for i in range(len(pesos[0])):
    file.write(str(i) + " ")
    for j in range(len(pesos)):
        file.write(str(pesos[j][i], 6) + " ")

    file.write("\n")
file.close()

# Se guarda la red neuronal entrenada
nn.save(red, "Pesos/NN.txt", salida)
