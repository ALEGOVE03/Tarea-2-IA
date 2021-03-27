import matplotlib.pyplot as plt
import math



def Sigmoide(a):
    
    y=1/(1+math.exp(-a))

    return y


def Relu(a):

    if(0<a):
        y=a
    else:
        y=0  
    return y


def Tanh(a):
    
    y=math.tanh(a)

    return y



#graficar y probar las funciones 
ent=[]
sal=[]

i=-10
while (i<11):
    ent.append(i)
    sal.append(Tanh(i))
    i=i+0.01

plt.plot(ent,sal)
plt.show()
