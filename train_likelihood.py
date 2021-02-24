import os, sys, argparse, importlib, time
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from tensorflow.python.ops.math_ops import reduce_euclidean_norm
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score



from tqdm import tqdm

log = True

if log: 
    import wandb 
    run = wandb.init(project="likelihood_angle", entity="johannbs")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_probability as tfp

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model


from spektral.data import DisjointLoader
from spektral.transforms.one_hot import OneHotLabels

file_path = osp.dirname(osp.realpath(__file__))

################################################
# Setup Deafult Variabls                       # 
################################################
learning_rate = 1e-4
batch_size    = 1024
epochs        = 100
early_stop    = True
patience      = 20
model_name    = "GCN_likelihood3"


################################################
# Setup Hyperparameters                        # 
################################################
hidden_states = 32
forward       = False
dropout       = 0.25
loss_method   = "likelihood_covariant_unitvectors"
n_neighbors   = 6 # SKRIV SELV IND


if log:
    # Declare for log
    wandb.config.hidden_states = hidden_states
    wandb.config.forward = forward
    wandb.config.dropout = dropout
    wandb.config.learning_rate = learning_rate
    wandb.config.batch_size = batch_size
    wandb.config.loss_func = loss_method
    wandb.config.n_neighbors = n_neighbors



################################################
# Get model and data                           # 
################################################
from models.GCN_likelihood import model
model = model(n_out = 6, hidden_states=hidden_states, forward=forward, dropout = dropout)
# model = load_model(osp.join(file_path, "models", "saved_models", "MessPass1"))



from data.graph_w_edge2_vects import graph_w_edge2
dataset = graph_w_edge2()
idx_lists = dataset.index_lists


# Split data
dataset_train = dataset[idx_lists[0]]
dataset_val   = dataset[idx_lists[1]]
dataset_test  = dataset[idx_lists[2]]


# Make loaders
loader_train  = DisjointLoader(dataset_train, epochs = epochs, batch_size = batch_size)
loader_test   = DisjointLoader(dataset_test , epochs = 1,      batch_size = batch_size)

save_path     = osp.join(file_path, "models", "saved_models", model_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

################################################
# Loss and Optimation                          # 
################################################
import loss_functions

loss_func     = getattr(loss_functions, loss_method)

opt           = Adam(learning_rate = learning_rate)

def angle(pred, true):
    return tf.math.acos(
        tf.clip_by_value(
            tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1)),
            -1., 1.)
        )


def metrics(y_reco, y_true):
    # Angle metric
    angle_resi = 180 / np.pi * angle(y_reco[:, :3], y_true[:, :3])

    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(u_angle.numpy())


def lr_schedule(epochs = epochs, initial = learning_rate, decay = 0.9):
    n = 1
    lr = initial
    yield lr
    while n < 4:
        lr *= 2
        n  += 1
        yield lr
    while True:
        lr *= decay
        n  += 1 
        yield lr

        


################################################
# TF - functions                               # 
################################################

@tf.function(input_signature = loader_train.tf_signature(), experimental_relax_shapes = True)
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
        targets     = tf.cast(targets, tf.float32)
        loss        = loss_func(targets, predictions)
        loss       += sum(model.losses)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss




@tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
def test_step(inputs, targets):
    predictions = model(inputs, training = False)
    targets     = tf.cast(targets, tf.float32) 
    out         = loss_func(targets, predictions)

    return predictions, targets, out


def validation(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000 
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)
    
    y_reco  = tf.concat(prediction_list, axis = 0)
    y_true  = tf.concat(target_list, axis = 0)

    metric = metrics(y_true, y_reco)

    # auc     = metric.result()
    loss    = loss_func(y_true, y_reco)

    return loss, metric


def test(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0).numpy()




################################################
# Training                                     # 
################################################

current_batch = 0
current_epoch = 1
loss          = 0
lowest_loss   = 9999999
early_stop_counter    = 0

pbar          = tqdm(total = loader_train.steps_per_epoch, position = 0, leave = True)
start_time    = time.time()
lr_gen        = lr_schedule()
learning_rate = next(lr_gen)


for batch in loader_train:
    inputs, targets  = batch
    inputs[0][:, :3] = inputs[0][:, :3] / 1000
    out              = train_step(inputs, targets)
    loss            += out

    current_batch  += 1
    pbar.update(1)
    pbar.set_description(f"Epoch {current_epoch} / {epochs}; Avg_loss: {loss / current_batch:.6f}")


    if current_batch == loader_train.steps_per_epoch:
        
        print(f"Epoch {current_epoch} of {epochs} done in {time.time() - start_time:.2f} seconds using learning rate: {learning_rate:.2E}")
        print(f"Avg loss of train: {loss / loader_train.steps_per_epoch:.6f}")

        loader_val    = DisjointLoader(dataset_val, epochs = 1,      batch_size = batch_size)
        val_loss, val_metric = validation(loader_val)

        print(f"Avg loss of validation: {val_loss:.6f}, Angle_metric: {val_metric:.6f}")

        if log:
            wandb.log({"Train Loss":      loss / loader_train.steps_per_epoch,
                    "Validation Loss": val_loss, 
                    "Angle Metric":    val_metric})


        if val_loss < lowest_loss:
            early_stop_counter = 0
            lowest_loss        = val_loss
        else:
            early_stop_counter += 1
        
        if early_stop and (early_stop_counter >= patience):
            model.save(save_path)
            print(f"Stopped training. No improvement was seen in {patience} epochs")
            break

        if current_epoch != epochs:
            pbar.n            = 0
            pbar.last_print_n = 0
            pbar.refresh()

        learning_rate = next(lr_gen)
        opt.learning_rate.assign(learning_rate)

        if current_epoch % 10 == 0:
            model.save(save_path)
            print("Model saved")

        loss            = 0
        start_time      = time.time()
        current_epoch  += 1
        current_batch   = 0


if log:
    run.finish()        
fig, ax = test(loader_test)
fig.savefig(f"model_tests/{model_name}_test.pdf")