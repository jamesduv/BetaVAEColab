#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:27:49 2020

@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""

import tensorflow as tf
import numpy as np

def vae_gaussian_latent(y_true, y_pred, sigma, mu ):
    '''Compute vae loss assuming gaussian latents and priors.

    Use mse term for reconstruction loss

    Args:
        sigma (tensor) : std deviation for latent dist.
        mu (tensor) : mean for latent dist.
        y_true (tensor) : the true, target output
        y_pred (tensor) : the predicted, target output

    Returns:
        loss (float) : the computed value of the loss function
    '''

    #KL divergence term
    dkl = -0.5 * tf.reduce_sum(1 + tf.math.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

    #reconstruction term
    ##TODO: Fix this, put in correct form
    recon_loss = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred), [1,2])

    #total loss
    loss = tf.math.reduce_mean(dkl + recon_loss)
    return loss, dkl, recon_loss


# def vae_gaussian_latent_wrapper(sigma, mu):
#     '''Wrapper for using vae loss function with Gaussian distributions

#     Args: 
#         sigma (tensor) : std deviation for latent dist.
#         mu (tensor) : mean for latent dist.

#     Returns:
#         loss (float) : The loss for the given model
#     '''
#     def vae_gaussian_latent(y_true, y_pred):
#         #KL divergence term
#         dkl = -0.5 * tf.reduce_sum(1 + tf.math.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

#         #reconstruction term
#         recon_mse = tf.keras.losses.MSE(y_true, y_pred)

#         #total loss
#         loss = tf.math.reduce_mean(dkl + recon_mse)
#         return loss
#     return vae_gaussian_latent

