import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import tanh
from spektral.layers import MessagePassing, GlobalAvgPool

# from sonnet import Linear
# from sonnet.nets import MLP



class model(Model):

    def __init__(self, hidden_states = 64,  **kwargs):
        # Encode X and E
        super().__init__()
        self.encode_x1 = Dense(hidden_states // 2)
        self.encode_x2 = Dense( hidden_states, activation = 'relu')
        self.encode_e1 = Dense(hidden_states // 2)
        self.encode_e2 = Dense( hidden_states, activation = 'relu')

        # self.MP_model1    = MP(n_out = hidden_states, hidden_states = hidden_states)
        self.mp_layers    = [MP(n_out = hidden_states, hidden_states = hidden_states) for i in range(3)]
        self.norm_layers  = [BatchNormalization() for i in range(3)]
        self.norm_decode  = BatchNormalization()
        self.pool         = GlobalAvgPool()
        self.decode       = MLP(output = hidden_states, hidden = hidden_states * 2, layers = 4)
        self.out1         = Dense(hidden_states // 2)
        self.out2         = Dense(7)


    
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

      forwards  = tf.gather(x[:, 3], send) < tf.gather(x[:, 3], receive)

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

    def __init__(self, n_out, hidden_states):
        super().__init__()
        self.n_out = n_out
        self.hidden_states = hidden_states
        self.message_mlp = MLP(hidden_states * 2, hidden = hidden_states * 4, layers = 2)
        self.update_mlp  = MLP(hidden_states * 1, hidden = hidden_states * 2, layers = 2)

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, e)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages)

        # Update
        # upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings)

        return output

    def message(self, x, a, e):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlp(out)
        return out
    
    def update(self, embeddings):
        out = self.update_mlp(embeddings)
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


    def call(self, inputs):
        return self.mlp(inputs)