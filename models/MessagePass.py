import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.activations import tanh
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.sparse import SparseTensor, eye, add
from spektral.layers import MessagePassing, GlobalAvgPool

# from sonnet import Linear
# from sonnet.nets import MLP



class model(Model):

    def __init__(self, hidden_states = 64,  dropout = 0, forward = False):
        # Encode X and E
        super().__init__()
        self.forward = forward
        self.encode_x1 = Dense(hidden_states // 2)
        self.encode_x2 = Dense( hidden_states, activation = 'relu')
        self.encode_e1 = Dense(hidden_states // 2)
        self.encode_e2 = Dense( hidden_states, activation = 'relu')

        # self.MP_model1    = MP(n_out = hidden_states, hidden_states = hidden_states)
        self.mp_layers    = [MP(n_out = hidden_states, dropout = dropout, hidden_states = hidden_states) for i in range(3)]
        self.norm_layers  = [BatchNormalization() for i in range(3)]
        self.norm_decode  = BatchNormalization()
        self.pool         = GlobalAvgPool()
        self.decode       = MLP(output = hidden_states, hidden = hidden_states * 2, layers = 4)
        self.out1         = Dense(hidden_states)
        self.out2         = Dense(1, activation = "sigmoid")

        self.activation   = LeakyReLU(0.15)


    
    def call(self, inputs, training = False):
        x, a, i    = inputs
        a, e       = self.generate_edge_features(x, a)
        x          = self.encode_x1(x)
        x          = self.encode_x2(x)
        e          = self.encode_e1(e)
        e          = self.encode_e2(e)
        for MP, norm in zip(self.mp_layers, self.norm_layers):
            x      = norm(x, training = training)
            x      = MP([x, a, e])
        x          = self.norm_decode(x, training = training)
        x          = self.pool([x, i])
        x          = self.decode(x)
        x          = tanh(self.out1(x))
        x          = self.out2(x)
        return x

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      if self.forward:
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



class MP(MessagePassing):

    def __init__(self, n_out, hidden_states, dropout = 0):
        super().__init__()
        self.n_out = n_out
        self.hidden_states = hidden_states
        self.message_mlp = MLP(hidden_states * 2, hidden = hidden_states * 4, layers = 2, dropout = dropout)
        self.update_mlp  = MLP(hidden_states * 1, hidden = hidden_states * 2, layers = 2, dropout = dropout)

    def propagate(self, x, a, e=None, training = False, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, e, training = training)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, training = training)

        # Update
        # upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, training = training)

        return output

    def message(self, x, a, e, training = False):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlp(out, training = training)
        return out
    
    def update(self, embeddings, training = False):
        out = self.update_mlp(embeddings, training = training)
        return out

class MLP(Model):
    def __init__(self, output, hidden=256, layers=2, batch_norm=True,
                 dropout=0.0, activation='relu', final_activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output, activation = activation))
            if dropout > 0:
                self.mlp.add(Dropout(dropout))


    def call(self, inputs, training = False):
        return self.mlp(inputs, training = training)