import tensorflow as tf


class Autoencoder(object):
    """
    Implementation of an autoencoder as a neural network used to learn an
    efficient data encoding in an unsupervised manner.
    """
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, activation):
        """
        :param batch_size:
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param activation:
        """
        assert input_dim == output_dim, 'The input and output dimension should be the same'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        if activation.lower() in ['relu']:
            self.activation = tf.nn.relu
            self.dactivation = lambda x: (tf.sign(x) + 1.) / 2.
        elif activation.lower() in ['tanh']:
            self.activation = tf.tanh
            self.dactivation = lambda x: 4. / (tf.exp(x) + tf.exp(-x)) ** 2
        elif activation.lower() in ['identity']:
            self.activation = lambda x: x
            self.dactivation = lambda x: tf.ones_like(x)
        elif activation.lower() in ['sigmoid', 'sigd']:
            self.activation = tf.sigmoid
            self.dactivation = lambda x: tf.exp(x) / (1. + tf.exp(x)) ** 2
        elif activation.lower() in ['negative']:
            self.activation = lambda x: -x
            self.dactivation = lambda x: -tf.ones_like(x)
        else:
            raise ValueError('unrecognized activation function')

        self.inputs = tf.placeholder(tf.float32, shape = [batch_size, input_dim])
        self.step_size = tf.placeholder(tf.float32, shape = [])

        with tf.variable_scope('autoencoder') as scope:
            self.encoder_weight = tf.get_variable(
                    name='encoder_weight', shape = [input_dim, hidden_dim],
                    initializer=tf.truncated_normal_initializer(stddev = 0.02),
                    dtype = tf.float32)
            self.decoder_weight = tf.get_variable(
                    name = 'decoder_weight', shape = [hidden_dim, output_dim],
                    initializer = tf.truncated_normal_initializer(stddev = 0.02),
                    dtype = tf.float32)

        self.projection = tf.matmul(self.inputs, self.encoder_weight)
        self.encode = self.activation(self.projection)
        self.decode = tf.matmul(self.encode, self.decoder_weight)

        error = self.decode - self.inputs
        self.loss = tf.reduce_mean(tf.reduce_sum(error ** 2, axis=1)) / 2

        # TODO give the update formula to update both matrices

        updated_encoder = None      # Need modification
        updated_decoder = None      # Need modification

        raise NotImplementedError('You should write your code HERE')

        self.encoder_update = tf.assign(self.encoder_weight, updated_encoder)
        self.decoder_update = tf.assign(self.decoder_weight, updated_decoder)

    def model_init(self,):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(
                var_list=[self.encoder_weight, self.decoder_weight],
                max_to_keep=1)

    def model_end(self,):
        self.sess.close()

    def save_model(self, file2dump):
        self.saver.save(self.sess, file2dump)

    def load_model(self, file2load):
        self.saver.restore(self.sess, file2load)

    def train(self, data_batch, step_size):
        """
        training a model

        :param data_batch: of shape [batch_size, input_dim]
        :param step_size: float, step size
        """
        train_dict = {self.inputs: data_batch, self.step_size: step_size}
        loss, _, _ = self.sess.run(
                [self.loss, self.encoder_update, self.decoder_update],
                feed_dict = train_dict)
        return loss

    def test(self, data_batch):
        """
        test a model

        :param data_batch: of shape [batch_size, input_dim]
        """
        test_dict = {self.inputs: data_batch,}
        loss = self.sess.run(self.loss, feed_dict=test_dict)
        return loss

    def compress(self, data_batch):
        """
        compress an image

        :param data_batch: of shape [batch_size, input_dim]
        """
        compress_dict = {self.inputs: data_batch}
        encode = self.sess.run(self.encode, feed_dict = compress_dict)
        return encode

    def reconstruct(self, data_batch):
        """
        reconstruct the image
        """
        reconstruct_dict = {self.inputs: data_batch}
        decode = self.sess.run(self.decode, feed_dict = reconstruct_dict)
        return decode
