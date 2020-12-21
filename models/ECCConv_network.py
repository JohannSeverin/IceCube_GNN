import os
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalSumPool 

from tensorflow.keras import Model, Input



hidden_states = 30
layers        = 2



class model(Model):
    def __init__(self):
        super().__init__()
        # Define input layers
        # self.x_in = Input(shape = (5, None), name = "Input of X")
        # self.a_in = Input(shape = (None, ), sparse = True, name = "Input of A")
        # self.e_in = Input(shape = (5, None), name = "Input of E")
        # self.i_in = Input(shape = (),name = "Input of I" )

        # Define layers of the model
        self.ECC1  = ECCConv(32, [hidden_states] * 2, n_out = hidden_states, activation = "relu")
        self.ECC2  = ECCConv(64, [hidden_states] * 2, n_out = hidden_states, activation = "relu")
        self.GCN   = GCNConv(32, activation = "relu")
        self.Pool  = GlobalSumPool()

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.ECC1([x, a, e])
        x = self.ECC2([x, a, e])
        x = self.GCN([x, a])
        x = self.Pool([x, i])
        return x

        
        