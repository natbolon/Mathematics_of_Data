import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from util import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=100,
        help='the dimension of hidden neurons, default = 100')
    parser.add_argument('--activation', type=str, default='relu',
        help='the activation function to use, default = "relu", allowed = ["relu", "tanh", "identity", "sigmoid", "negative"]')
    parser.add_argument('--load_file', type=str, default="model.ckpt",
        help='the file to be loaded, default = "model.ckpt"')
    parser.add_argument('--image_shown', type=int, default=4,
        help='the number of images shown, default = 4')

    args = parser.parse_args()

    input_dim = 28 * 28
    output_dim = 28 * 28
    hidden_dim = args.hidden_dim
    activation = args.activation
    load_file = args.load_file
    image_shown = args.image_shown

    # obtain the dataset
    mnist = input_data.read_data_sets('./data', one_hot=False)

    model = Autoencoder(image_shown, input_dim, hidden_dim, output_dim, activation)
    model.model_init()
    model.load_model(load_file)

    data_batch, label_batch = mnist.test.next_batch(image_shown)
    data_batch = data_batch.reshape(image_shown, -1)
    reconstruct_batch = model.reconstruct(data_batch)

    original_batch = data_batch.reshape(-1, 28, 28)
    reconstruct_batch = reconstruct_batch.reshape(-1, 28, 28)

    for plot_idx in range(image_shown):
        plt.subplot(2, image_shown, 2 * plot_idx + 1)
        plt.imshow(original_batch[plot_idx], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, image_shown, 2 * plot_idx + 2)
        plt.imshow(reconstruct_batch[plot_idx], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.show()
