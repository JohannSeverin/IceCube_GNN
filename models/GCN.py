import os
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalMaxPool 

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import tanh



hidden_states = 64


# Probably needs regularization, but first step is just to fit, then we will regularize.

class model(Model):
    def __init__(self, n_out = 6):
        super().__init__()
        # Define layers of the model
        self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states, hidden_states], n_out = hidden_states, activation = "relu")
        self.GCN1    = GCNConv(hidden_states, activation = "relu")
        self.GCN2    = GCNConv(hidden_states * 2, activation = "relu")
        self.GCN3    = GCNConv(hidden_states * 4, activation = "relu")
        self.GCN4    = GCNConv(hidden_states * 8, activation = "relu")
        self.Pool    = GlobalMaxPool()
        # self.decode  = [Dense(hidden_states * 4)] + [Dense(hidden_states * 2)] + [Dense(hidden_states) for i in range(layers - 2 )]
        self.decode  = [Dense(size * hidden_states) for size in [8, 4, 2, 1, 1, 1]]
        self.d2      = Dense(n_out)

    def call(self, inputs, training = False):
        x, a, i = inputs
        e       = self.generate_edge_features(x, a)
        x = self.ECC1([x, a, e])
        x = self.GCN1([x, a])
        x = self.GCN2([x, a])
        x = self.GCN3([x, a])
        x = self.GCN4([x, a])
        x = self.Pool([x, i])
        for decode_layer in self.decode:
          x = tanh(decode_layer(x))
        x = self.d2(x)
        return x

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]

      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))

      e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)


      return e


        
        