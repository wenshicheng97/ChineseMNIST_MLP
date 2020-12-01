import numpy as np
import pandas as pd
import readfile
from neuralNetwork import NeuralNetwork
import train_test
import evaluation

if __name__ == '__main__':
    input_nodes = 4096
    hidden_nodes = 1000
    output_nodes = 15

    learning_rate = 0.1

    rounds = 1

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    n.load()

    train_test.train_neuralnetwork(n, rounds)

    real, pred = train_test.predict_neuralnetwork(n)

    n.save()

    precision, recall, f1_macro, f1_micro, kappa = evaluation.evaluate(real, pred)
