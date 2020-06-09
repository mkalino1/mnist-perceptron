import random
from math_functions import *
import numpy as np


class Perceptron():

    def __init__(self, sizes):
        """Inizjalizacja. """
        self.sizes = sizes

        # lista losowych macierzy bias dla kazdej wartwy nauronow z wyjatkiem pierwszej
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # lista losowych macierzy krawedzi laczacych sasiadujace warstwy neuronow
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def train(self, training_data, test_data, epochs, mini_batch_size, eta):
        """Trenowanie sieci przy uzyciu algorytmu stochastycznego najszybszego spadku
            przy użyciu mini-batch"""

        for j in range(epochs):
            # Shuffle, aby w kazdej epoce nie bylo takiego samego podzialu na minibatche
            random.shuffle(training_data)

            # Dokonaj podzialu na minibatche
            mini_batches = []
            for x in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[x: x+mini_batch_size])

            # Aktualizuj weights i biases na podstawie danych z kazdego minibatcha osobno
            for mini_batch in mini_batches:
                # Listy zbierajace sumaryczny gradient po wszystkich obrazkach w mini batchu
                gradient_biases = [np.zeros(b.shape) for b in self.biases]
                gradient_weights = [np.zeros(w.shape) for w in self.weights]

                # Dla kazdego obrazka znajdujacego sie w mini-batch uwzglednij zwiazany z nim gradient
                for image, label in mini_batch:
                    delta_gradient_biases, delta_gradient_weights = self.backpropagation(image, label)
                    gradient_biases = [grad + delta for grad, delta in zip(gradient_biases, delta_gradient_biases)]
                    gradient_weights = [grad + delta for grad, delta in zip(gradient_weights, delta_gradient_weights)]

                # Zaktualizuj weights i biases zgodnie z algorytmem stochastycznego najszybszego spadku uwzgledniajac odpowiednie learning rate
                self.weights = [w - (eta / len(mini_batch)) * grad for w, grad in zip(self.weights, gradient_weights)]
                self.biases = [b - (eta / len(mini_batch)) * grad for b, grad in zip(self.biases, gradient_biases)]

            # Sprawdz postepy algorytmu na końcu każdej epoki na danych testowych
            correct_answers = self.how_many_correct_answers(test_data)
            print(f"Epoch number {j}: {correct_answers}/{len(test_data)} correct answers")


    def backpropagation(self, image, label):
        """Oblicz gradient zwiazany z pojedynczym obrazkiem metodą backpropagation"""

        # listy, ktore beda zawieraly gradient obliczony za pomoca backpropagation
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # lista activations bedzie przechowywala wszystkie kolejne wartstwy aktywacji
        # wprowadzamy pierwsza warstwe aktywacji, czyli piksele naszego obrazka
        activations = [image]

        # analogiczna lista, ktora bedzie przechowywala kolejne warstwy efektów sumowania (przed zastosowaniem funkcji aktywacji)
        sumator_layers = []

        # zapelnienie tych dwóch list idąc wprzód sieci
        current_activation = image
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_activation)+b
            sumator_layers.append(z)
            current_activation = sigmoid(z)
            activations.append(current_activation)

        # pochodna funkcji kosztu, trzeba pamietac tez o pochodnej funkcji sigmoidalnej
        # poniewaz ostatnia warstwa sieci również ma sigmuoidalna funkcje aktywacji
        delta = cost_function_derivative(activations[-1], label) * sigmoid_derivative(sumator_layers[-1])

        # zapełniamy bezpośrednio liste gradient dla ostatniej warstwy
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())

        # przechodzimy od przedostatniej warstwy sieci do pierwszej wyliczajac gradient
        for i in range(2, len(self.sizes)):
            z = sumator_layers[-i]
            sig_der = sigmoid_derivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sig_der

            gradient_b[-i] = delta
            gradient_w[-i] = np.dot(delta, activations[-i-1].transpose())

        return gradient_b, gradient_w

    def feedforward(self, a):
        """Poslugujac sie rachunkiem na macierzach zwraca wyjscie sieci dla zadanego wejscia sieci"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def how_many_correct_answers(self, test_data):
        """Zwraca liczbe poprawnych klasyfikiacji sieci dla danych testowych.
        Przez klasyfikacje sieci nalezy rozumiec cyfre powiazana z neuronem o najwiekszej aktywacji w wyjsciu sieci"""
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        correct_answers = sum(int(x == y) for x, y in test_results)
        return correct_answers


