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
            self.activation=lambda x: torch.clamp(x, min = 0.)
            self.dactivation=lambda x: (torch.sign(x) + 1) / 2
        elif activation.lower() in ['tanh']:
            self.activation=lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
            self.dactivation=lambda x: 4 / (torch.exp(x) + torch.exp(-x)) ** 2
        elif activation.lower() in ['identity']:
            self.activation=lambda x: x
            self.dactivation=lambda x: torch.ones_like(x)
        elif activation.lower() in ['sigmoid', 'sigd']:
            self.activation=lambda x: 1. / (1. + torch.exp(-x))
            self.dactivation=lambda x: torch.exp(x) / (1 + torch.exp(x)) ** 2
        elif activation.lower() in ['negative']:
            self.activation=lambda x: -x
            self.dactivation=lambda x: -torch.ones_like(x)
        else:
            raise ValueError('unrecognized activation function')

    def train(self, data_batch, step_size):
        """
        training a model

        :param data_batch: of shape [batch_size, input_dim]
        :param step_size: float, step size
        """

        projection = torch.matmul(data_batch, self.encoder_weight)
        encode = self.activation(projection) #shape [data_batch, hidden_size]
        decode = torch.matmul(encode, self.decoder_weight)  # of shape [batch_size, output_dim]

        hidden_dim = projection.size()[1]
        input_dim = decode.size()[1]

        # Reshape tensors
        projection = projection.unsqueeze_(2).permute(1, 2, 0)  # shape [hidden_size, 1, batch_size]
        encode = encode.unsqueeze_(2).permute(1, 2, 0) #shape [hidden_size, 1, batch_size]
        decode = decode.unsqueeze_(2).permute(1, 2, 0) #shape [output_dim, 1, batch_size]
        self.encoder_weight = self.encoder_weight.unsqueeze_(2).repeat(1, 1, self.batch_size) #shape [input_size, hidden_size, batch_size]
        self.decoder_weight = self.decoder_weight.unsqueeze_(2).repeat(1, 1, self.batch_size)  # shape [input_size, hidden_size, batch_size]
        data_batch = data_batch.unsqueeze_(2).permute(1, 2, 0)

        error = decode - data_batch
        loss = torch.mean(torch.sum(error ** 2, dim=2)) / 2


        # TODO calculate the gradient and update the weight

        # Get data dimensions
        hidden_dim = encode.size()[0]
        batch_size = data_batch.size()[2]

        # Derivate activation function
        dact_proj = self.dactivation(projection)


        # Compute gradient w.r.t. W2
        error_rep = error.expand(error.size()[0], hidden_dim, error.size()[2])

        encode_rep = encode.permute(1, 0, 2)
        encode_rep = encode_rep.expand(input_dim, encode_rep.size()[1], encode_rep.size()[2])

        grad_dec = -2*torch.mul(error_rep,encode_rep)
        grad_dec = grad_dec.permute(1, 0, 2)



        # Compute gradient w.r.t. W1
        # Expand along columns the derivative of the projection
        # Final shape [Hidden_size, Input_size, Batch_size]
        dact_proj = dact_proj.expand(dact_proj.size()[0], input_dim, dact_proj.size()[2])

        # Expand input. Final shape [Hidden_size, Input_size, Batch_size]
        input_exp = data_batch.permute(1, 0, 2)
        input_exp = input_exp.expand(hidden_dim, input_exp.size()[1], input_exp.size()[2])

        grad_enc = torch.mul(dact_proj, input_exp)

        #Compute W2 components on grad_enc
        # Pick only the first matrix of the tensor
        w2 = self.decoder_weight[:, : ,0]

        # Sum W2 along rows
        ones = torch.ones((input_dim, 1))
        sum_weights = torch.matmul(w2, ones)
        sum_weights.unsqueeze_(2)

        # Expand. Final shape [Hidden_size, Input_size, Batch_size]
        sum_weights = sum_weights.expand(sum_weights.size()[0], input_dim, batch_size)

        grad_enc = torch.mul(grad_enc, sum_weights)

        error_rep = error_rep.permute(1,0,2)

        grad_enc = torch.mul(grad_enc, error_rep)
        grad_enc = grad_enc.permute(1,0,2)
        grad_enc = -2*grad_enc


        # Update W2 --- SEEMS TO BE WORKING!

        # TODO MAKE AVERAGE ALONG DEPTH! DO NOT TAKE FIRST SLIDE!
        self.decoder_weight += step_size * grad_dec
        self.decoder_weight = torch.mean(self.decoder_weight, dim=2)


        #Update W1
        #self.encoder_weight += step_size*grad_enc
        #self.encoder_weight = torch.mean(self.encoder_weight, dim=2)
        self.encoder_weight = self.encoder_weight[:,:,0]

        print loss
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
