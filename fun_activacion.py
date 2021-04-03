import matplotlib.pyplot as plt
import math

def Sigmoide(x):
    y = 1 / (1 + math.exp(-x))
    return y

def Relu(a):
    if(0 < a):
        y = a
    else:
        y = 0
    return y

def Tanh(a):
    y = math.tanh(a)
    return y

#graficar y probar las funciones
ent = []
sal = []

i=-10
while (i < 11):
    ent.append(i)
    sal.append(Sigmoide(i))
    i= i + 0.1

plt.plot(ent,sal)
plt.show()
