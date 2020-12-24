import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from spektral.layers import MessagePassing, GlobalAvgPool

from sonnet import Linear
from sonnet.nets import MLP



class model(Model):

    def __init__(self, hidden_states = 64,  **kwargs):
        # Encode X and E
        super().__init__()
        self.encode_x = Dense( hidden_states)
        self.encode_e = Dense( hidden_states)

        # self.MP_model1    = MP(n_out = hidden_states, hidden_states = hidden_states)
        self.mp_layers    = [MP(n_out = hidden_states, hidden_states = hidden_states) for i in range(3)]
        self.pool         = GlobalAvgPool()
        self.decode       = MLP(output = hidden_states, hidden = hidden_states * 2, layers = 2)
        self.out          = Dense(1)


    
    def call(self, inputs):
        x, a, e, i = inputs
        x          = self.encode_x(x)
        e          = self.encode_e(e)
        for MP in self.mp_layers:
            x      = MP([x, a, e])
        x          = self.pool(x)
        x          = self.decode(x)
        x          = self.out(x)
        return x



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
                 dropout=0.0, activation='prelu', final_activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output))


    def call(self, inputs):
        return self.mlp(inputs)