import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from util import Autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_num', type=int, default=50,
        help='the number of epochs, default = 50')
    parser.add_argument('--hidden_dim', type=int, default=100,
        help='the dimension of hidden neurons, default = 100')
    parser.add_argument('--activation', type=str, default='relu',
        help='the activation function to use, default = "relu", allowed = ["relu", "tanh", "identity", "sigmoid", "negative"]')
    parser.add_argument('--step_size', type=float, default=0.01,
        help='the step size, default = 0.01')
    parser.add_argument('--save_file', type=str, default="model.ckpt",
        help='the file to restore the model, default="model.ckpt"')

    args = parser.parse_args()

    input_dim = 28 * 28
    output_dim = 28 * 28
    batch_size = 100
    epoch_num = args.epoch_num
    hidden_dim = args.hidden_dim
    activation = args.activation
    step_size = args.step_size
    save_file = args.save_file

    mnist = input_data.read_data_sets('./data', one_hot = True)
    train_total_batch = int(mnist.train.num_examples / batch_size)
    test_total_batch = int(mnist.test.num_examples / batch_size)

    model = Autoencoder(batch_size, input_dim, hidden_dim, output_dim, activation)
    model.model_init()

    train_loss_list = []
    test_loss_list = []
    for epoch_idx in range(epoch_num):
        loss_list = []
        for batch_idx in range(train_total_batch):
            data_batch, label_batch = mnist.train.next_batch(batch_size)
            data_batch = data_batch.reshape(batch_size, -1)
            loss = model.train(data_batch, step_size)
            loss_list.append(loss)
        loss_mean = np.mean(loss_list)
        train_loss_list.append(loss_mean)
        print('Train loss after epoch %d: %1.3e'%(epoch_idx + 1, loss_mean))

        loss_list = []
        for batch_idx in range(test_total_batch):
            data_batch, label_batch = mnist.test.next_batch(batch_size)
            data_batch = data_batch.reshape(batch_size, -1)
            loss = model.test(data_batch)
            loss_list.append(loss)
        loss_mean = np.mean(loss_list)
        test_loss_list.append(loss_mean)
        print('Test loss after epoch %d: %1.3e'%(epoch_idx + 1, loss_mean))

    model.save_model(save_file)
    print('Model saved in %s'%save_file)

    plt.plot(np.arange(1, epoch_num + 1), train_loss_list, label='train loss', color='r')
    plt.plot(np.arange(1, epoch_num + 1), test_loss_list, label='test loss', color='b')
    plt.legend()
    plt.show()
