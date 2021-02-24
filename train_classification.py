import os, sys, argparse, importlib, time, pickle
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from tensorflow.python.ops.math_ops import reduce_euclidean_norm
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

import wandb 
wandb.init(project="stopped_muons", entity="johannbs")


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
learning_rate = 2.5e-4
batch_size    = 1024
epochs        = 100
early_stop    = True
patience      = 3
model_name    = "GCN_classification0"


################################################
# Setup Hyperparameters                        # 
################################################
hidden_states = 128
forward       = False
dropout       = 0.5
# loss_method   = "loss_func_linear_angle"
n_neighbors   = 9 # SKRIV SELV IND



# Declare for log
wandb.config.hidden_states = hidden_states
wandb.config.forward = forward
wandb.config.dropout = dropout
wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
# wandb.config.loss_func = loss_method
wandb.config.n_neighbors = n_neighbors



################################################
# Get model and data                           # 
################################################
# from models.GCN_stopped_muons import model
# model = model(hidden_states=hidden_states, forward=forward, dropout = dropout)
model = load_model(osp.join(file_path, "models", "saved_models", "GCN_classification0"))



from data.graph_stopped_muons import graph_w_edge2
dataset = graph_w_edge2()
idx_lists = dataset.index_lists


# Split data
dataset_train = dataset[idx_lists[0]]
dataset_val   = dataset[idx_lists[1]]
dataset_test  = dataset[idx_lists[2]]

# for set in [dataset_train, dataset_val, dataset_test]:
#     set.apply(OneHotLabels(labels = (0, 1)))

# Make loaders
loader_train  = DisjointLoader(dataset_train, epochs = epochs, batch_size = batch_size)
loader_test   = DisjointLoader(dataset_test , epochs = 1,      batch_size = batch_size)

save_path     = osp.join(file_path, "models", "saved_models", model_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

################################################
# Loss and Optimation                          # 
################################################
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy

loss_func     = BinaryCrossentropy()

opt           = Adam(learning_rate = learning_rate)

metric        = BinaryAccuracy()


def lr_schedule(epochs = epochs, initial = learning_rate, decay = 0.9):
    n = 1
    lr = initial
    yield lr
    while n < 3:
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
        # targets          = tf.squeeze(targets) 
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)
    
    y_reco  = tf.concat(prediction_list, axis = 0)
    y_true  = tf.concat(target_list, axis = 0)
    auc     = roc_auc_score(y_true.numpy(), y_reco.numpy())

    metric.reset_states()
    metric.update_state(y_true, y_reco)


    acc     = metric.result()

    # auc     = metric.result()
    loss    = loss_func(y_true, y_reco)

    return loss, auc, acc


def test(loader):
    loss = 0
    prediction_list, target_list = [], []
    N_nodes = []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000
        # targets          = tf.squeeze(targets)
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        y, idx, count = tf.unique_with_counts(batch[0][2])

        N_nodes.append(count)
        prediction_list.append(predictions)
        target_list.append(targets)

        

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0).numpy()
    N_nodes = tf.concat(N_nodes, axis = 0).numpy()

    pickle.dump({'true': y_true, 'y_reco': y_reco, 'N': N_nodes}, open(osp.join("model_tests", "classification_params"), 'wb'))





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
    inputs, targets = batch
    inputs[0][:, :3] = inputs[0][:, :3] / 1000
    # targets          = tf.squeeze(targets)  
    out              = train_step(inputs, targets)
    loss            += out

    current_batch  += 1
    pbar.update(1)
    pbar.set_description(f"Epoch {current_epoch} / {epochs}; Avg_loss: {loss / current_batch:.6f}")


    if current_batch == loader_train.steps_per_epoch:
        
        print(f"Epoch {current_epoch} of {epochs} done in {time.time() - start_time:.2f} seconds using learning rate: {learning_rate:.2E}")
        print(f"Avg loss of train: {loss / loader_train.steps_per_epoch:.6f}")

        loader_val    = DisjointLoader(dataset_val, epochs = 1,      batch_size = batch_size)
        val_loss, val_auc, val_acc = validation(loader_val)

        print(f"Avg loss of validation: {val_loss:.6f}, AUC: {val_auc:.6f}, Accuracy: {val_acc:.6f}")


        wandb.log({"Train Loss":      loss / loader_train.steps_per_epoch,
                   "Validation Loss": val_loss, 
                   "AUC":             val_auc, 
                   "Accuracy":        val_acc})


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
            pbar          = tqdm(total = loader_train.steps_per_epoch, position = 0, leave = True)

        learning_rate = next(lr_gen)
        opt.learning_rate.assign(learning_rate)

        if current_epoch % 10 == 0:
            model.save(save_path)
            print("Model saved")

        loss            = 0
        start_time      = time.time()
        current_epoch  += 1
        current_batch   = 0

        
test(loader_test)
# fig.savefig(f"model_tests/{model_name}_test.pdf")