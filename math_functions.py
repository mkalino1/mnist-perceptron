# Funkcje matematyczne, ktorych uzywa perceptron

import numpy as np


def sigmoid(z):
    """Funkcja sigmuoidalna"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """Pochodna funkcji sigmuoidalnej"""
    return sigmoid(z)*(1-sigmoid(z))


def cost_function_derivative(output_activations, label_digit):
    """Pochodna funkcji kosztu (bledu sredniokwadratowego)"""

    # zamiana na postac wektorowa, zeby uzywac rachunku macierzy
    label_vector = np.zeros((10, 1))
    label_vector[label_digit] = 1.0

    return 2 * (output_activations-label_vector)
