import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pickle
import numpy as np

from torch.autograd import Variable


def obtain_dataloader(batch_size):
    """
    obtain the data loader
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class Autoencoder(nn.Module):
    """
    Autoencoder
    """

    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, activation):
        """
        specify an autoencoder
        """
        assert input_dim == output_dim, 'The input and output dimension should be the same'
        self.encoder_weight = torch.randn([input_dim, hidden_dim]) * 0.02
        self.decoder_weight = torch.randn([hidden_dim, output_dim]) * 0.02
        self.batch_size = batch_size

        if activation.lower() in ['relu']:
            self.activation = lambda x: torch.clamp(x, min=0.)
            self.dactivation = lambda x: (torch.sign(x) + 1) / 2
        elif activation.lower() in ['tanh']:
            self.activation = lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
            self.dactivation = lambda x: 4 / (torch.exp(x) + torch.exp(-x)) ** 2
        elif activation.lower() in ['identity']:
            self.activation = lambda x: x
            self.dactivation = lambda x: torch.ones_like(x)
        elif activation.lower() in ['sigmoid', 'sigd']:
            self.activation = lambda x: 1. / (1. + torch.exp(-x))
            self.dactivation = lambda x: torch.exp(x) / (1 + torch.exp(x)) ** 2
        elif activation.lower() in ['negative']:
            self.activation = lambda x: -x
            self.dactivation = lambda x: -torch.ones_like(x)
        else:
            raise ValueError('unrecognized activation function')

    def train(self, data_batch, step_size):
        """
        training a model

        :param data_batch: of shape [batch_size, input_dim]
        :param step_size: float, step size
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)  # shape [data_batch, hidden_size]
        decode = torch.matmul(encode, self.decoder_weight)  # of shape [batch_size, output_dim]
        dact_proj = self.dactivation(projection)

        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim=0)) / 2

        # Get data dimensions
        input_dim = data_batch.size()[1]
        hidden_dim = encode.size()[1]
        batch_size = data_batch.size()[0]

        # Reshape tensors
        encode = encode.unsqueeze_(2)  # shape [batch_size, hidden_size, 1]
        err = error.unsqueeze_(2).permute(0, 2, 1)  # shape [batch_size, 1, input_size]
        err = err.expand(batch_size, hidden_dim, input_dim)  # shape [batch_size, hidden_size, input_size]

        # Gradient w.r.t. W2
        grad_dec = torch.mean(-1 * torch.mul(err, encode.expand(batch_size, hidden_dim, input_dim)),
                              dim=0)  # shape[input_size, hidden_size]

        # Compute gradient w.r.t. W1

        """
        
        # Step 1. -2*W2*e.t
        grad_enc = -2*torch.matmul(self.decoder_weight.permute(2, 0, 1), error.permute(2, 0, 1))

        # Step 2. Element-wise multiplication by dact_proj
        grad_enc = torch.mul(grad_enc, dact_proj.permute(2, 0, 1))
        grad_enc = grad_enc.expand(grad_enc.size()[0], grad_enc.size()[1], hidden_dim)

        # Step 3. Matrix multiplication by Input
        data = data_batch.permute(2, 0, 1)
        grad_enc = torch.matmul(data.expand(data.size()[0], data.size()[1], hidden_dim), grad_enc)
        
        """

        """
        # Step 1.
        e = error.permute(2, 1, 0)
        w2 = self.decoder_weight.permute(2, 1, 0)
        grad_enc = torch.matmul(e,w2)
        grad_enc = grad_enc.expand(grad_enc.size()[0], input_dim, grad_enc.size()[2])

        # Step 2
        der = dact_proj.permute(2, 1, 0)
        der = der.expand(der.size()[0], input_dim, der.size()[2])
        grad_enc = torch.mul(grad_enc, der)

        # Step 3
        data = data_batch.permute(2, 0, 1)
        data = data.expand(data.size()[0], data.size()[1], hidden_dim)
        grad_enc = torch.mul(grad_enc, data)

        # Step 4
        grad_enc = -2*grad_enc.permute(1, 2, 0)
        """

        # Step 0. Generate tensors of correct shape
        W2 = self.decoder_weight.unsqueeze(2)
        inp = data_batch.unsqueeze(2)
        dact_proj = dact_proj.unsqueeze_(2).permute(0, 2, 1)

        # Step 1. Element-wise multiplication input-derivate(projection)
        grad_enc = torch.mul(inp.permute(0, 1, 2).expand(batch_size, input_dim, hidden_dim),
                             dact_proj.expand(batch_size, input_dim, hidden_dim))

        # Step 2. Matrix multiplication W2-error
        M = torch.matmul(W2.permute(2, 0, 1).expand(batch_size, hidden_dim, input_dim), error).permute(0, 2, 1).expand(
            batch_size, input_dim, hidden_dim)

        # Step 3.
        grad_enc = torch.mean(-torch.mul(grad_enc, M), dim=0)

        # Update W2
        self.decoder_weight += step_size * grad_dec

        # Update W1
        self.encoder_weight += step_size * grad_enc

        return float(loss)

    def test(self, data_batch):
        """
        test and calculate the reconstruction loss
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim=1)) / 2

        return loss

    def compress(self, data_batch):
        """
        compress a data batch
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)

        return np.array(encode)

    def reconstruct(self, data_batch):
        """
        reconstruct the image
        """
        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection)
        decode = torch.matmul(encode, self.decoder_weight)

        return np.array(decode)

    def save_model(self, file2dump):
        """
        save the model
        """
        pickle.dump(
            [np.array(self.encoder_weight), np.array(self.decoder_weight)],
            open(file2dump, 'wb'))

    def load_model(self, file2load):
        """
        load the model
        """
        encoder_weight, decoder_weight = pickle.load(open(file2load, 'rb'))
        self.encoder_weight = torch.FloatTensor(encoder_weight)
        self.decoder_weight = torch.FloatTensor(decoder_weight)
