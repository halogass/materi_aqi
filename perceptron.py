import random
import os

lr = 1
bias = 1
weights = [random.random(),random.random(),random.random()]
print("Initial weights: ", weights)

def trainPerceptron(input1, input2, output) :
    outputP = input1*weights[0] + input2*weights[1] + bias*weights[2]
    if outputP > 0 : #activation function (here Heaviside)
        outputP = 1
    else :
        outputP = 0
    error = output - outputP
    weights[0] += error * input1 * lr
    weights[1] += error * input2 * lr
    weights[2] += error * bias * lr
    #print(error)

for i in range(500000) :
    value1 = random.randrange(1000000) / 1000000
    value2 = random.randrange(1000000) / 1000000
    if value1 < 0.45 and value2 < 0.65:
        trainPerceptron(value1, value2, 1)
    else:
        trainPerceptron(value1, value2, 0)
    
print("Final weights: ", weights)


x = float(input())
y = float(input())
outputP = x * weights[0] + y * weights[1] + bias * weights[2]
if outputP > 0 : #activation function
   outputP = 1
else :
   outputP = 0
print(x, "and", y, "is : ", outputP)
