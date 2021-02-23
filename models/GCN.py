
        
import os
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool
from spektral.utils import gcn_filter

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor



hidden_states = 64
activation = LeakyReLU(alpha = 0.1)

# Probably needs regularization, but first step is just to fit, then we will regularize.

class model(Model):
    def __init__(self, n_out = 7, hidden_states = 64, forward = False, dropout = 0):
        super().__init__()
        self.forward = forward
        # Define layers of the model
        self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states, hidden_states], n_out = hidden_states, activation = "relu")
        self.GCN1    = GCNConv(hidden_states, activation = "relu")
        self.GCN2    = GCNConv(hidden_states * 2, activation = "relu")
        self.GCN3    = GCNConv(hidden_states * 4, activation = "relu")
        # self.GCN4    = GCNConv(hidden_states * 8, activation = "relu")
        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.decode  = [Dense(size * hidden_states) for size in [12, 12, 8]]
        self.drop_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]

        self.small_layers = [Dense(hidden_states // 4) for i in range(n_out)]
        self.out          = [Dense(1) for i in range(n_out)]


    def call(self, inputs, training = False):
        x, a, i = inputs
        a, e    = self.generate_edge_features(x, a)
        x = self.ECC1([x, a, e])
        # a = gcn_filter(a)
        x = self.GCN1([x, a])
        x = self.GCN2([x, a])
        x = self.GCN3([x, a])
        # x = self.GCN4([x, a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)
        for decode_layer, norm_layer, drop_layer in zip(self.decode, self.norm_layers, self.drop_layers):
          x = drop_layer(x, training = training)
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)
        outs = []
        for s, o in zip(self.small_layers, self.out):
          outs.append(o(s(x)))
        return tf.concat(outs, axis = 1)



        return x

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
      if self.forward == True:
        forwards  = tf.gather(x[:, 3], send) <= tf.gather(x[:, 3], receive)

        send    = tf.cast(send[forwards], tf.int64)
        receive = tf.cast(receive[forwards], tf.int64)

        a       = SparseTensor(indices = tf.stack([send, receive], axis = 1), values = tf.ones(tf.shape(send), dtype = tf.float32), dense_shape = tf.cast(tf.shape(a), tf.int64))

      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))

      e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)

      return a, e


        
        