from random import random

import numpy as np

#Clase para definir una capa de la red neuronal (
class Layer:
    def _init_(self, x, w, b):
        self.x = np.array(x)
        self.w = np.array(w)
        self.b = np.array(b)

    def calcular(self):
        z = np.dot(self.w, self.x) + self.b
        return z


class RedNeuronal:
    def _init_(self, alpha):
        self.w1 = np.array([[random(),random()], [random(), random()], [random(), random()]])
        self.w2 = np.array([[random(), random(), random()], [random(), random(), random()]])
        self.b1 = np.array([[random()], [random()], [random()]])
        self.b2 = np.array([[random()], [random()]])

        self.alpha = alpha

    def backwardPropagation(self, x1, x2, y1, y2):
        x = np.array([[x1], [x2]])
        y = np.array([[y1], [y2]])
        layer1 = Layer(x, self.w1, self.b1)

        z = layer1.calcular()

        a1 = sigmoid(z[0, 0])
        a2 = sigmoid(z[1, 0])
        a3 = sigmoid(z[2, 0])
        af1 = np.array([[a1], [a2], [a3]])

        layer2 = Layer(af1, self.w2, self.b2)

        z2 = layer2.calcular()

        a4 = sigmoid(z2[0, 0])
        a5 = sigmoid(z2[1, 0])
        A2 = np.array([[a4], [a5]])

        #Termina feed forward
        delta_z2 = A2 - y

        delta_w2 = np.dot(delta_z2, af1.T)
        self.w2 = self.w2 - alpha * delta_w2
        self.b2 = self.b2 - alpha * delta_z2

        delta_z1 = ( np.dot(self.w2.T, delta_z2) * (af1 * (1 - af1)) )
        delta_w1 = np.dot(delta_z1, x.T)
        self.w1 = self.w1 - alpha * delta_w1
        self.b1 = self.b1 - alpha * delta_z1

        return A2

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a



alpha = 0.25
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
e = [[0, 0], [1, 0], [1, 0], [0, 1]]
ejemplo = RedNeuronal(alpha)

for i in range(5000):
    for inputs, output in zip(x,e):
        salida = ejemplo.backwardPropagation(inputs[0], inputs[1], output[0], output[1])
	
    if i == 4999:
    	print("Salida para [" + str(inputs[0]) + ", " + str(inputs[1]) +"]: ")
    	print(salida)
    	print("Aproximadamente: [" + str(round(salida[0,0])) + ", " + str(round(salida[1,0])) + "]")
    	print()	

