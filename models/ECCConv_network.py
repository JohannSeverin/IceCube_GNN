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
layers        = 4



class model(Model):
    def __init__(self):
        super().__init__()
        # Define layers of the model
        self.ECC1    = ECCConv(hidden_states, [hidden_states] * 3, n_out = hidden_states, activation = "relu")
        self.GCN1    = GCNConv(hidden_states, activation = "relu")
        self.GCN2    = GCNConv(hidden_states * 2, activation = "relu")
        self.GCN3    = GCNConv(hidden_states * 4, activation = "relu")
        self.Pool    = GlobalMaxPool()
        self.decode  = [Dense(hidden_states * 4)] + [Dense(hidden_states * 2)] + [Dense(hidden_states) for i in range(layers - 2 )]
        self.d2      = Dense(1)

    def call(self, inputs, training = False):
        x, a, e, i = inputs
        x = self.ECC1([x, a, e])
        x = self.GCN1([x, a])
        x = self.GCN2([x, a])
        x = self.GCN3([x, a])
        x = self.Pool([x, i])
        for decode_layer in self.decode:
          x = tanh(decode_layer(x))
        x = self.d2(x)
        return x

        
        