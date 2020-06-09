from perceptron import Perceptron
from mnist_loading import load_data
import time


def main():

    training_data, test_data = load_data()

    layers = [784, 16, 10]
    epochs = 20
    minibatch_size = 20
    learning_rate = 4.0

    model = Perceptron(layers);

    start = time.time()
    print("Starting training...")
    model.train(training_data , test_data, epochs, minibatch_size, learning_rate)
    print("Training complete")
    end = time.time()

    print(f"Training time: {end - start}")

if __name__ == '__main__':
    main()
