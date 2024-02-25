"""
Created on Sat Dec 12 11:43:32 2020

@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import pickle, os, time
import numpy as np
import tensorflow as tf

import util
import network_util as nu

def swish(x):
    return (x*tf.keras.activations.sigmoid(x))

tf.keras.utils.get_custom_objects().update({'swish': swish})   
tf.keras.utils.get_custom_objects().update({'swish_fn': swish})

class vae_v1():
    '''variational autoencoder - conditionals as dense NN's

    Encoder: q
    Decoder: p

    Uses a variable number of dense layers for generating mu, sigma_sq for
    sampling from latent distribution, all dense layers have same size,
    equal to 2 x latent_dim

    All neural network layers use same activation function, except (optionally)
    the encoder and decoder outputs, which may use linear output

    Args:
        opt (dict)      : options for network construction and training
        data (dict)     : optional, provide training data

    Attributes:
        encoder (tf.keras.Model) : 
        decoder (tf.keras.Model) : 

    '''

    def __init__(self, opt, data={}):
        self.opt    = opt
        self.data   = data

        #set the loss function, based on opt
        self.f_loss = self.set_loss()
        self.beta_loss = opt['beta_loss']

        #set kernel_regularizer, acitvation function(s), and optimizer
        self.kernel_regularizer     = nu.set_kernel_regularizer(opt)
        self.activation             = nu.set_activation(opt)
        self.optimizer              = nu.set_optimizer(opt)

        #filenames for saving training loss and weights
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')


        self.fn_weights_enc_best = os.path.join(self.opt['save_dir'],
                                            'weights.encoder.best.h5')
        self.fn_weights_enc_end = os.path.join(self.opt['save_dir'],
                                           'weights.encoder.end.h5')
        self.fn_weights_dec_best = os.path.join(self.opt['save_dir'],
                                            'weights.decoder.best.h5')
        self.fn_weights_dec_end = os.path.join(self.opt['save_dir'],
                                           'weights.decoder.end.h5')

        self.fn_weights_best = {'encoder':self.fn_weights_enc_best,
                               'decoder':self.fn_weights_dec_best}
        self.fn_weights_end = {'encoder':self.fn_weights_enc_end,
                               'decoder':self.fn_weights_dec_end}
        self.build_model()

    def predict(self, x):
        '''Predict on provided data x, return all output from 'call_model' in 
        dictionary'''

        if x.dtype is not float:
            x = x.astype('float')

        mu, log_sigma, sigma, z, y_pred = self.call_model(x)
        data = {'mu':mu, 'log_sigma':log_sigma, 'z':z, 'y_pred':y_pred}
        return data


    def call_model(self, x):
        '''Feed forward, full model

        Args:
            x (array like) : input, numpy array or tf.Tensor

        Returns:
            mu (tf.Tensor)      : mean of sampled latent dist q(z|x)
            log_sigma (tf.Tensor) : log of variance of sampled latent dist q(z|x)
            sigma (tf.Tensor)   : variance of sampled latent dist q(z|x)
            z (tf.Tensor)       : encoder raw output, q(z|x)
            y_pred (tf.Tensor)  : decoder output, p(x|z)
        '''

        mu, log_var = self.encode(x)
        var = tf.exp(log_var)
        z = self.reparameterize(mean = mu, var = var)
        y_pred = self.decoder(z)
        return mu, log_var, var, z, y_pred

    def build_model(self):
        '''Build encoder and decoder models using the options defined in 
        self.opt and class attributes.'''

        self.layer_names_encoder = []
        self.int_models = {}
        # input_encoder = tf.keras.Input(shape=(self.opt['ny'] * self.opt['nx'],
        #                                       self.opt['n_input']),
        #                                       name='input_encoder')
        input_encoder = tf.keras.Input(shape=(self.opt['ny'], self.opt['nx'],
                                              self.opt['n_input']),
                                              name='input_encoder')
        name = 'flatten_encoder'
        self.layer_names_encoder.append(name)
        output = tf.keras.layers.Flatten(name=name)(input_encoder)

        inp_shape = output.shape

        print('Building encoder')
        for iDense in range(self.opt['n_layers_encoder']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            self.layer_names_encoder.append(name)
            units = self.opt['n_nodes_encoder'][iDense]
            activation = self.activation
            
            # if iDense == 0:
            #     output = tf.keras.layers.Dense(units = units,
            #                                    activation = activation,
            #                                    kernel_regularizer = self.kernel_regularizer,
            #                                    name = name)(input_encoder)
                
                
            # else:
            output = tf.keras.layers.Dense(units = units,
                                           activation = activation,
                                           kernel_regularizer = self.kernel_regularizer,
                                           name = name)(output)
            if self.opt['is_batch_norm_encoder']:
                name = 'batch_norm_' + name
                self.layer_names_encoder.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)

        print('Encoder Output Layer')
        name = 'encoder_output'
        self.layer_names_encoder.append(name)
        units = 2 * self.opt['latent_dim']
        if self.opt['is_sigmoid_output_encoder']:
            activation = tf.keras.activations.sigmoid
        elif self.opt['is_linear_output_encoder']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)

        #make encoder, generate models for each level of output
        self.encoder = tf.keras.Model(inputs=[input_encoder], outputs=[output])
        self.int_models_encoder = {}
        for name in self.layer_names_encoder:
            cur_output = self.encoder.get_layer(name).output
            self.int_models_encoder[name] = tf.keras.Model(inputs=[input_encoder],
                                                  outputs=[cur_output])

        print('Building Decoder')
        self.layer_names_decoder = []
        input_decoder = tf.keras.Input(shape=(int(self.opt['latent_dim']),),
                                       name='input_decoder')

        output = None

        # Dense Layers Construction - decoder
        print('Constructing the Dense Layers')
        for iDense in range(self.opt['n_layers_encoder']):
            idx =self.opt['n_layers_encoder'] - iDense - 1
            print('Decoder Dense Layer {:1.0f}'.format(iDense))
            name = 'dense_' + str(iDense) + '_decoder'
            self.layer_names_decoder.append(name)
            units = self.opt['n_nodes_encoder'][idx]
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               activation = self.activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(input_decoder)
            else:
                output = tf.keras.layers.Dense(units = units,
                                               activation = self.activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)

        print('Decoder Output Layer')
        name = 'decoder_output'
        self.layer_names_decoder.append(name)
        # units = self.opt['ny'] * self.opt['nx']
        units = inp_shape[1]
        if self.opt['is_sigmoid_output_decoder']:
            activation = tf.keras.activations.sigmoid
        elif self.opt['is_linear_output_decoder']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output = tf.keras.layers.Dense(units = units,
                               activation = activation,
                               kernel_regularizer = self.kernel_regularizer,
                               name = name)(output)
        name = 'reshape_decoder'
        self.layer_names_decoder.append(name)
        output = tf.keras.layers.Reshape((self.opt['ny'], self.opt['nx'], 1),
                                         name=name)(output)


        #make decoder, generate models for each level of output
        self.decoder = tf.keras.Model(inputs=[input_decoder], outputs=[output])
        self.int_models_decoder = {}
        for name in self.layer_names_decoder:
            cur_output = self.decoder.get_layer(name).output
            self.int_models_decoder[name] = tf.keras.Model(inputs=[input_decoder], outputs=[cur_output])

        print('Models Created')
        self.encoder.summary()
        self.decoder.summary()

    def train_model(self, data = None):
        '''Train the model using VAE loss function. If no data is provided, 
        use the data provided upon class instance construction

        Args:
            data (dict) : training/validation data
        '''

        if data is not None:
            self.data = data
        #convert data to float
        # for k,v in self.data.items():
        #     self.data[k] = v.astype(np.float)

        self.create_train_log_KL()
        n_batches = int(np.ceil(self.opt['n_train'] / self.opt['batch_size']))
        batch_size = self.opt['batch_size']
        ny = self.opt['ny']
        nx = self.opt['nx']
        n_epochs = self.opt['epochs']

        #there are updated and used to shuffle minibatches
        self.idx_train = np.arange(self.data['x_train'].shape[0])
        self.idx_val  = np.arange(self.data['x_val'].shape[0])

        training_loss = []
        val_loss_store = []
        val_best_store = []
        dkl_train = []
        dkl_val_store = []
        recon_train = []
        recon_val_store = []
        for epoch in np.arange(n_epochs):
            #get shuffled dataset every epoch
            dset = self.shuffle_dset(n_batches, batch_size)

            #loop over mini batches, back propagate, apply gradients
            batch_loss = []
            dkl_batch = []
            recon_batch = []
            for iBatch, (xb_train, yb_train) in enumerate(dset):

                loss, dkl, recon_loss = self.train_on_batch(xb_train, yb_train)

                batch_loss.append(tf.math.reduce_mean(loss, keepdims=True))
                dkl_batch.append(tf.math.reduce_mean(dkl, keepdims=True))
                recon_batch.append(tf.math.reduce_mean(recon_loss, keepdims=True))

            training_loss.append(np.mean(np.array(batch_loss)))
            dkl_train.append(np.mean(np.array(dkl_batch)))
            recon_train.append(np.mean(np.array(recon_batch)))

            #validation data
            mu, log_var, var, z, y_pred = self.call_model(\
                                                      self.data['x_val'])

            loss_val, dkl_val, recon_loss_val = self.f_loss(self.data['y_val'],
                                                    y_pred, var, log_var, mu)

            val_loss_store.append(loss_val.numpy().mean())
            dkl_val_store.append(dkl_val.numpy().mean())
            recon_val_store.append(recon_loss_val.numpy().mean())

            if epoch == 0:
                val_best_cur = loss_val.numpy()

            self.print_train_val(epoch+1, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                print('Saving best weights')
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)

            # self.update_training(epoch, training_loss[-1], val_loss_store[-1])
            self.update_training_KL(epoch, training_loss[-1],
                                    val_loss_store[-1], dkl_train[-1],
                                    dkl_val_store[-1],
                                    recon_train[-1], recon_val_store[-1])

            #save end weights and z
            if np.mod(epoch, self.opt['n_epoch_save']) == 0:
                print('Saving end weights')
                self.save_weights(self.fn_weights_end)

                #TODO: Add z to this, so evolution through training can be observed
                # message = 'Saving z to : {}'.format(self.fn_z_end)
                # print(message)
                # pickle.dump(z_var, open(self.fn_z_end, 'wb'))

    def train_on_batch(self, xb_train, yb_train):
        with tf.GradientTape() as tape:
            mu, log_var, var, z, y_pred = self.call_model(xb_train)
    
            loss, dkl, recon_loss = self.f_loss(\
                                    yb_train, y_pred, var, log_var, mu)


        grads = tape.gradient(loss, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads,
                                            tape.watched_variables()))
        return loss, dkl, recon_loss


    def save_weights(self, fn_weights):
        self.encoder.save_weights(fn_weights['encoder'])
        self.decoder.save_weights(fn_weights['decoder'])

    def load_weights(self, fn_weights):
        self.encoder.load_weights(fn_weights['encoder'])
        self.decoder.load_weights(fn_weights['decoder'])

    def shuffle_dset(self, n_batches, batch_size):
        '''Generate a list of tuples (x,y) over shuffled data in self.data'''
        np.random.shuffle(self.idx_train)
        np.random.shuffle(self.idx_val)
        dset = []
        for iBatch in np.arange(n_batches):
            idx1 = iBatch * batch_size
            idx2 = idx1 + batch_size
            x_cur = self.data['x_train'][idx1:idx2,:,:,:]
            y_cur = self.data['y_train'][idx1:idx2,:,:,:]
            dset.append((x_cur, y_cur))
        return dset

    def set_loss(self):
        f_all = {'gaussian':self.loss_bvae_gaussian,
                 'bernoulli':self.loss_bvae_bernoulli}
        dist = self.opt['output_distribution']
        return f_all[dist]

    def loss_bvae_gaussian(self, y_true, y_pred, var, log_var, mu):
        '''Compute beta-VAE loss with gaussian output distribution

        Gaussian prior on latent distribution'''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var, 1)
    
        #reconstruction term
        mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred),
                                    [1,2])
        # recon_loss = 0.5 * tf.math.log((2 * np.pi * mse)+1)
        recon_loss = mse

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def loss_bvae_bernoulli(self, y_true, y_pred, var, log_var, mu):
        '''Compute beta-VAE loss with bernoulli output distribution

        Gaussian prior on latent distribution'''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var, 1)
    
        #reconstruction term
        f_recon =  tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                      reduction=tf.keras.losses.Reduction.NONE)
        # recon_loss_int = f_recon(y_true, y_pred)
        recon_loss_int = f_recon(y_true, y_pred)
        recon_loss = tf.math.abs(tf.reduce_mean(recon_loss_int, [1,2]))

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def loss_beta_vae(self, y_true, y_pred, var, log_var, mu):
        '''Compute vae loss assuming gaussian prior/latent p(z), p(z|x).
       
        Args:
            sigma (tensor) : std deviation for latent dist.
            mu (tensor) : mean for latent dist.
            y_true (tensor) : the true, target output
            y_pred (tensor) : the predicted, target output
    
        Returns:
            loss (float) : the computed value of the loss function
        '''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var, 1)
    
        #reconstruction term
        if self.opt['output_distribution'] == 'gaussian':
            mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred),
                                    [1,2])
            recon_loss = 0.5 * tf.math.log((2 * np.pi * mse)+1)

        elif self.opt['output_distribution'] == 'bernoulli':
            f_recon =  tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                      reduction=tf.keras.losses.Reduction.NONE)
            recon_loss_int = f_recon(y_true, y_pred)
            recon_loss = tf.reduce_mean(recon_loss_int, [1,2])

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def encode(self, x):
        '''Pass the data x through the encoder network. Return the mean
        and log variance of the  latent distribution'''

        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        if self.opt['is_sigmoid_var_encoder']:
            if not self.opt['is_sigmoid_output_encoder']:
                logvar = tf.keras.activations.sigmoid(logvar)

        return mean, logvar

    def reparameterize(self, mean, var):
        '''Reparameterize the latent output using a sample from epsilon'''

        shape = [self.opt['latent_dim']]
        eps = tf.random.normal(shape=shape)
        z = mean + tf.math.sqrt(var) * eps
        return z

    def create_train_log(self):
        '''Create training log file, headers: (epoch, loss, val_loss)'''
        f = open(self.fn_train,'a+')
        header =('epoch','loss','val_loss\n')
        f.write(','.join(header))
        f.close()

    def update_training(self, epoch, training_loss, val_loss_store):
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss_store) + '\n')))
        f.close()

    def create_train_log_KL(self):
        f = open(self.fn_train,'a+')
        header = ('epoch', 'loss', 'val_loss', 'KL', 'val_KL',
                  'recon', 'recon_val\n')
        f.write(','.join(header))
        f.close()

    def update_training_KL(self, epoch, training_loss, val_loss_store,
                              dkl_train, dkl_val, recon_train, recon_val):
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss_store), str(dkl_train),
                          str(dkl_val), str(recon_train),
                          str(recon_val)+ '\n')))
        f.close()

    def print_train_no_val(self, epoch, n_epochs, loss):
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}'.format(epoch, n_epochs, loss)
        print(message)

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)


class vae_v2():
    '''variational autoencoder - conditionals as dense NN's

    Encoder: q
    Decoder: p

    Uses a variable number of dense layers for generating mu, sigma_sq for
    sampling from latent distribution, all dense layers have same size,
    equal to 2 x latent_dim

    All neural network layers use same activation function, except (optionally)
    the encoder and decoder outputs, which may use linear output

    Args:
        opt (dict)      : options for network construction and training
        data (dict)     : optional, provide training data

    Attributes:
        encoder (tf.keras.Model) : 
        decoder (tf.keras.Model) : 

    '''

    def __init__(self, opt, data={}):
        self.opt    = opt
        self.data   = data

        #set the loss function, based on opt
        self.f_loss = self.set_loss()
        self.beta_loss = opt['beta_loss']

        #set kernel_regularizer, acitvation function(s), and optimizer
        self.kernel_regularizer     = nu.set_kernel_regularizer(opt)
        self.activation             = nu.set_activation(opt)
        self.optimizer              = nu.set_optimizer(opt)

        #filenames for saving training loss and weights
        self.fn_train = os.path.join(self.opt['save_dir'], 'training.csv')


        self.fn_weights_enc_best = os.path.join(self.opt['save_dir'],
                                            'weights.encoder.best.h5')
        self.fn_weights_enc_end = os.path.join(self.opt['save_dir'],
                                           'weights.encoder.end.h5')
        self.fn_weights_dec_best = os.path.join(self.opt['save_dir'],
                                            'weights.decoder.best.h5')
        self.fn_weights_dec_end = os.path.join(self.opt['save_dir'],
                                           'weights.decoder.end.h5')

        self.fn_weights_best = {'encoder':self.fn_weights_enc_best,
                               'decoder':self.fn_weights_dec_best}
        self.fn_weights_end = {'encoder':self.fn_weights_enc_end,
                               'decoder':self.fn_weights_dec_end}
        self.build_model()

    def predict(self, x):
        '''Predict on provided data x, return all output from 'call_model' in 
        dictionary'''

        if x.dtype is not float:
            x = x.astype('float')

        mu, log_sigma, sigma, z, y_pred = self.call_model(x)
        data = {'mu':mu, 'log_sigma':log_sigma, 'z':z, 'y_pred':y_pred}
        return data


    def call_model(self, x):
        '''Feed forward, full model

        Args:
            x (array like) : input, numpy array or tf.Tensor

        Returns:
            mu (tf.Tensor)      : mean of sampled latent dist q(z|x)
            log_sigma (tf.Tensor) : log of variance of sampled latent dist q(z|x)
            sigma (tf.Tensor)   : variance of sampled latent dist q(z|x)
            z (tf.Tensor)       : encoder raw output, q(z|x)
            y_pred (tf.Tensor)  : decoder output, p(x|z)
        '''

        mu, log_var = self.encode(x)
        var = tf.exp(log_var)
        z = self.reparameterize(mean = mu, var = var)
        y_pred = self.decoder(z)
        return mu, log_var, var, z, y_pred

    def build_model(self):
        '''Build encoder and decoder models using the options defined in 
        self.opt and class attributes.'''

        self.layer_names_encoder = []
        self.int_models = {}
        # input_encoder = tf.keras.Input(shape=(self.opt['ny'] * self.opt['nx'],
        #                                       self.opt['n_input']),
        #                                       name='input_encoder')
        input_encoder = tf.keras.Input(shape=(self.opt['ny'], self.opt['nx'],
                                              self.opt['n_input']),
                                              name='input_encoder')
        name = 'flatten_encoder'
        self.layer_names_encoder.append(name)
        output = tf.keras.layers.Flatten(name=name)(input_encoder)

        inp_shape = output.shape

        print('Building encoder')
        for iDense in range(self.opt['n_layers_encoder']):
            print("Hidden Layer %d" % (iDense))
            name = 'dense_' + str(iDense)
            self.layer_names_encoder.append(name)
            units = self.opt['n_nodes_encoder'][iDense]
            activation = self.activation
            
            # if iDense == 0:
            #     output = tf.keras.layers.Dense(units = units,
            #                                    activation = activation,
            #                                    kernel_regularizer = self.kernel_regularizer,
            #                                    name = name)(input_encoder)
                
                
            # else:
            output = tf.keras.layers.Dense(units = units,
                                           activation = activation,
                                           kernel_regularizer = self.kernel_regularizer,
                                           name = name)(output)
            if self.opt['is_batch_norm_encoder']:
                name = 'batch_norm_' + name
                self.layer_names_encoder.append(name)
                output = tf.keras.layers.BatchNormalization(name=name)(output)

        print('Encoder Output Layers')
        name = 'encoder_output_mean'
        self.layer_names_encoder.append(name)
        units = self.opt['latent_dim']
        if self.opt['is_sigmoid_output_encoder']:
            activation = tf.keras.activations.sigmoid
        elif self.opt['is_linear_output_encoder']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output_mean = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)

        name = 'encoder_output_logvar'
        self.layer_names_encoder.append(name)
        units = self.opt['latent_dim']
        if self.opt['is_sigmoid_output_encoder']:
            activation = tf.keras.activations.sigmoid
        elif self.opt['is_linear_output_encoder']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output_logvar = tf.keras.layers.Dense(units = units,
                                       activation = activation,
                                       kernel_regularizer = self.kernel_regularizer,
                                       name = name)(output)




        #make encoder, generate models for each level of output
        self.encoder = tf.keras.Model(inputs=[input_encoder],
                                      outputs=[output_mean, output_logvar ])
        self.int_models_encoder = {}
        for name in self.layer_names_encoder:
            cur_output = self.encoder.get_layer(name).output
            self.int_models_encoder[name] = tf.keras.Model(inputs=[input_encoder],
                                                  outputs=[cur_output])

        print('Building Decoder')
        self.layer_names_decoder = []
        input_decoder = tf.keras.Input(shape=(int(self.opt['latent_dim']),),
                                       name='input_decoder')

        output = None

        # Dense Layers Construction - decoder
        print('Constructing the Dense Layers')
        for iDense in range(self.opt['n_layers_encoder']):
            idx =self.opt['n_layers_encoder'] - iDense - 1
            print('Decoder Dense Layer {:1.0f}'.format(iDense))
            name = 'dense_' + str(iDense) + '_decoder'
            self.layer_names_decoder.append(name)
            units = self.opt['n_nodes_encoder'][idx]
            if iDense == 0:
                output = tf.keras.layers.Dense(units = units,
                                               activation = self.activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(input_decoder)
            else:
                output = tf.keras.layers.Dense(units = units,
                                               activation = self.activation,
                                               kernel_regularizer = self.kernel_regularizer,
                                               name = name)(output)

        print('Decoder Output Layer')
        name = 'decoder_output'
        self.layer_names_decoder.append(name)
        # units = self.opt['ny'] * self.opt['nx']
        units = inp_shape[1]
        if self.opt['is_sigmoid_output_decoder']:
            activation = tf.keras.activations.sigmoid
        elif self.opt['is_linear_output_decoder']:
            activation = tf.keras.activations.linear
        else:
            activation = self.activation
        output = tf.keras.layers.Dense(units = units,
                               activation = activation,
                               kernel_regularizer = self.kernel_regularizer,
                               name = name)(output)
        name = 'reshape_decoder'
        self.layer_names_decoder.append(name)
        output = tf.keras.layers.Reshape((self.opt['ny'], self.opt['nx'], 1),
                                         name=name)(output)


        #make decoder, generate models for each level of output
        self.decoder = tf.keras.Model(inputs=[input_decoder], outputs=[output])
        self.int_models_decoder = {}
        for name in self.layer_names_decoder:
            cur_output = self.decoder.get_layer(name).output
            self.int_models_decoder[name] = tf.keras.Model(inputs=[input_decoder], outputs=[cur_output])

        print('Models Created')
        self.encoder.summary()
        self.decoder.summary()

    def train_model(self, data = None):
        '''Train the model using VAE loss function. If no data is provided, 
        use the data provided upon class instance construction

        Args:
            data (dict) : training/validation data
        '''

        if data is not None:
            self.data = data
        #convert data to float
        # for k,v in self.data.items():
        #     self.data[k] = v.astype(np.float)

        self.create_train_log_KL()
        n_batches = int(np.ceil(self.opt['n_train'] / self.opt['batch_size']))
        batch_size = self.opt['batch_size']
        ny = self.opt['ny']
        nx = self.opt['nx']
        n_epochs = self.opt['epochs']

        #there are updated and used to shuffle minibatches
        self.idx_train = np.arange(self.data['x_train'].shape[0])
        self.idx_val  = np.arange(self.data['x_val'].shape[0])

        training_loss = []
        val_loss_store = []
        val_best_store = []
        dkl_train = []
        dkl_val_store = []
        recon_train = []
        recon_val_store = []
        for epoch in np.arange(n_epochs):
            #get shuffled dataset every epoch
            dset = self.shuffle_dset(n_batches, batch_size)

            #loop over mini batches, back propagate, apply gradients
            batch_loss = []
            dkl_batch = []
            recon_batch = []
            for iBatch, (xb_train, yb_train) in enumerate(dset):

                loss, dkl, recon_loss = self.train_on_batch(xb_train, yb_train)

                batch_loss.append(tf.math.reduce_mean(loss, keepdims=True))
                dkl_batch.append(tf.math.reduce_mean(dkl, keepdims=True))
                recon_batch.append(tf.math.reduce_mean(recon_loss, keepdims=True))

            training_loss.append(np.mean(np.array(batch_loss)))
            dkl_train.append(np.mean(np.array(dkl_batch)))
            recon_train.append(np.mean(np.array(recon_batch)))

            #validation data
            mu, log_var, var, z, y_pred = self.call_model(\
                                                      self.data['x_val'])

            loss_val, dkl_val, recon_loss_val = self.f_loss(self.data['y_val'],
                                                    y_pred, var, log_var, mu)

            val_loss_store.append(loss_val.numpy().mean())
            dkl_val_store.append(dkl_val.numpy().mean())
            recon_val_store.append(recon_loss_val.numpy().mean())

            if epoch == 0:
                val_best_cur = loss_val.numpy()

            self.print_train_val(epoch+1, n_epochs, training_loss[-1],
                                 val_loss_store[-1])

            if val_loss_store[-1] <= val_best_cur:
                val_best_cur = val_loss_store[-1]
                print('Saving best weights')
                self.save_weights(self.fn_weights_best)

            val_best_store.append(val_best_cur)

            # self.update_training(epoch, training_loss[-1], val_loss_store[-1])
            self.update_training_KL(epoch, training_loss[-1],
                                    val_loss_store[-1], dkl_train[-1],
                                    dkl_val_store[-1],
                                    recon_train[-1], recon_val_store[-1])

            #save end weights and z
            if np.mod(epoch, self.opt['n_epoch_save']) == 0:
                print('Saving end weights')
                self.save_weights(self.fn_weights_end)

                #TODO: Add z to this, so evolution through training can be observed
                # message = 'Saving z to : {}'.format(self.fn_z_end)
                # print(message)
                # pickle.dump(z_var, open(self.fn_z_end, 'wb'))

    def train_on_batch(self, xb_train, yb_train):
        with tf.GradientTape() as tape:
            mu, log_var, var, z, y_pred = self.call_model(xb_train)
    
            loss, dkl, recon_loss = self.f_loss(\
                                    yb_train, y_pred, var, log_var, mu)


        grads = tape.gradient(loss, tape.watched_variables())
        self.optimizer.apply_gradients(zip(grads,
                                            tape.watched_variables()))
        return loss, dkl, recon_loss


    def save_weights(self, fn_weights):
        self.encoder.save_weights(fn_weights['encoder'])
        self.decoder.save_weights(fn_weights['decoder'])

    def load_weights(self, fn_weights):
        self.encoder.load_weights(fn_weights['encoder'])
        self.decoder.load_weights(fn_weights['decoder'])

    def shuffle_dset(self, n_batches, batch_size):
        '''Generate a list of tuples (x,y) over shuffled data in self.data'''
        np.random.shuffle(self.idx_train)
        np.random.shuffle(self.idx_val)
        dset = []
        for iBatch in np.arange(n_batches):
            idx1 = iBatch * batch_size
            idx2 = idx1 + batch_size
            x_cur = self.data['x_train'][idx1:idx2,:,:,:]
            y_cur = self.data['y_train'][idx1:idx2,:,:,:]
            dset.append((x_cur, y_cur))
        return dset

    def set_loss(self):
        f_all = {'gaussian':self.loss_bvae_gaussian,
                 'bernoulli':self.loss_bvae_bernoulli}
        dist = self.opt['output_distribution']
        return f_all[dist]

    def loss_bvae_gaussian(self, y_true, y_pred, var, log_var, mu):
        '''Compute beta-VAE loss with gaussian output distribution

        Gaussian prior on latent distribution'''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var, 1)
    
        #reconstruction term
        mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred),
                                    [1,2])
        # recon_loss = 0.5 * tf.math.log((2 * np.pi * mse)+1)
        recon_loss = mse

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def loss_bvae_bernoulli(self, y_true, y_pred, var, log_var, mu):
        '''Compute beta-VAE loss with bernoulli output distribution

        Gaussian prior on latent distribution'''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var,
                                  axis=1)
        #reconstruction term
        f_recon =  tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                      reduction=tf.keras.losses.Reduction.NONE)

        recon_loss_int = f_recon(y_true, y_pred)
        recon_loss = tf.reduce_sum(recon_loss_int, [1,2])

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def loss_beta_vae(self, y_true, y_pred, var, log_var, mu):
        '''Compute vae loss assuming gaussian prior/latent p(z), p(z|x).
       
        Args:
            sigma (tensor) : std deviation for latent dist.
            mu (tensor) : mean for latent dist.
            y_true (tensor) : the true, target output
            y_pred (tensor) : the predicted, target output
    
        Returns:
            loss (float) : the computed value of the loss function
        '''
        n_cases = y_true.shape[0]
        #KL divergence term - exact for Gaussian
        dkl = 0.5 * tf.reduce_sum(tf.math.square(mu) + var - 1 - log_var, 1)
    
        #reconstruction term
        if self.opt['output_distribution'] == 'gaussian':
            mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred),
                                    [1,2])
            recon_loss = 0.5 * tf.math.log((2 * np.pi * mse)+1)

        elif self.opt['output_distribution'] == 'bernoulli':
            f_recon =  tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                      reduction=tf.keras.losses.Reduction.NONE)
            recon_loss_int = f_recon(y_true, y_pred)
            recon_loss = tf.reduce_mean(recon_loss_int, [1,2])

        #total loss
        loss = tf.math.reduce_mean(self.beta_loss * dkl + recon_loss,
                                   keepdims=True)
        return loss, dkl, recon_loss

    def encode(self, x):
        '''Pass the data x through the encoder network. Return the mean
        and log variance of the  latent distribution'''

        mean, logvar = self.encoder(x)
        if self.opt['is_sigmoid_var_encoder']:
            if not self.opt['is_sigmoid_output_encoder']:
                logvar = tf.keras.activations.sigmoid(logvar)

        return mean, logvar

    def reparameterize(self, mean, var):
        '''Reparameterize the latent output using a sample from epsilon'''

        shape = [self.opt['latent_dim']]
        eps = tf.random.normal(shape=shape)
        z = mean + tf.math.sqrt(var) * eps
        return z

    def create_train_log(self):
        '''Create training log file, headers: (epoch, loss, val_loss)'''
        f = open(self.fn_train,'a+')
        header =('epoch','loss','val_loss\n')
        f.write(','.join(header))
        f.close()

    def update_training(self, epoch, training_loss, val_loss_store):
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss_store) + '\n')))
        f.close()

    def create_train_log_KL(self):
        f = open(self.fn_train,'a+')
        header = ('epoch', 'loss', 'val_loss', 'KL', 'val_KL',
                  'recon', 'recon_val\n')
        f.write(','.join(header))
        f.close()

    def update_training_KL(self, epoch, training_loss, val_loss_store,
                              dkl_train, dkl_val, recon_train, recon_val):
        f = open(self.fn_train,'a+')
        f.write(','.join((str(epoch), str(training_loss),
                          str(val_loss_store), str(dkl_train),
                          str(dkl_val), str(recon_train),
                          str(recon_val)+ '\n')))
        f.close()

    def print_train_no_val(self, epoch, n_epochs, loss):
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}'.format(epoch, n_epochs, loss)
        print(message)

    def print_train_val(self, epoch, n_epochs, loss, loss_val):
        message = 'Epoch [{:1.0f}/{:1.0f}]: Loss: {:1.4e}, Val_loss: {:1.4e}'\
                    .format(epoch, n_epochs, loss, loss_val)
        print(message)