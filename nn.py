import numpy as np
import math
import random
# Genetic Algorithm:
        # Create a population
        # Measure their fitness
        # Select fittest individuals
        # Crossover their genetics (Start with copy --> Crossover should mix random weight of first hidden layer matrix... see if better performance)
        # Mutate the resulting genes
        # Create new population and repeat
#Create initial number birds in memory to be drawn and each has their own distinct but random neural network
#Let each bird fly using neural network as a predictor for when they should jump at given input values
#Sort bird list by fitness scores and choose the highest ones?
#Mutate the genes at some probability around some mean zero
#Create new population.

#Forward propagation is an algorithm where you find the weighted sums of each weight matrix in respect to each layer and apply
#an activiation function to it. In this case, the sigmoid function.
#We want to find one output that is the probabilty that the bird jumps up or down, given some inputs. 

class NeuralNetwork():
    def __init__(self, *args):
        if len(args) == 3:
            (input_layer, hidden_layer, output_layer) = args

            self.input_layer = input_layer
            self.hidden_layer = hidden_layer
            self.output_layer = output_layer

            self.weights0 = random_matrix(hidden_layer, input_layer)
            self.weights1 = random_matrix(output_layer, hidden_layer) 

            self.bias0 = random_matrix(hidden_layer, 1)
            self.bias1 = random_matrix(output_layer, 1)

        elif len(args) == 4:
            (weight0, bias0, weight1, bias1) = args

            self.input_layer = len(weight0[0])
            self.hidden_layer = len(bias0)
            self.output_layer = len(bias1)

            self.weights0 = weight0
            self.weights1 = weight1
            self.bias0 = bias0
            self.bias1 = bias1
            

    def predict(self, input):
        #input layer
        a1 = input
        z2 = (self.weights0 @ a1) + self.bias0
        #hidden layer 1
        a2 = sigmoid(z2, False)
        z2 = (self.weights1 @ a2) + self.bias1

        #output layer
        a3 = sigmoid(z2, False)
        return a3

    def mutate(self):
        self.weights0 = my_map(mutate, self.weights0)
        self.weights1 = my_map(mutate, self.weights1)
        self.bias0 = my_map(mutate,self.bias0)
        self.bias1 = my_map(mutate, self.bias1)

def crossover(p1, p2):
    hid_len = len(p1.bias0)
    rand_hid = random.randint(0,hid_len)

    out_len = len(p1.bias1)
    rand_out = random.randint(0,out_len)
    
    p1w0 = p1.weights0[:rand_hid, :]
    p2w0 = p2.weights0[rand_hid:,:]

    w0 = np.vstack((p1w0, p2w0))
    b0 = p1.bias0
    w1 = p1.weights1 if rand_out == 0 else p2.weights1
    b1 = p1.bias1 
    return (w0,b0, w1,b1)
    
#10 percent chance of mutation
def mutate(x):
    rand = random.random()
    new_num = 0
    if(rand < .1):
        new_num = np.random.normal(0, .1)
    return x + new_num

def my_map(func, matrix):
    B = np.array( [func(x) for rows in matrix for x in rows])
    C = B.reshape((len(matrix),len(matrix[0])))
    return C

def random_matrix(row, col):
    return np.random.random((row, col)) * 2 - 1

def sigmoid(x, is_deriv):
    if is_deriv:
        return sigmoid(x, False) *  (1-sigmoid(x,False))
    else:
        return 1/(1+np.exp(-x)) 
