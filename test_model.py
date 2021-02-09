import os, sys, argparse, importlib, time

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_probability as tfp

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import load_model


file_path = osp.dirname(osp.realpath(__file__))

################################################
# Setup Deafult Variabls                       # 
################################################
learning_rate = 1e-3
batch_size    = 64
epochs        = 3
train_size    = 0.9



################################################
# Get Input from Terminal                      # 
################################################
parser = argparse.ArgumentParser(description = "Input for training file")
parser.add_argument("Model", action = "store", type = str)
parser.add_argument("Data", action = "store", type = str)
parser.add_argument("--train_size", action = "store", type = float, default = train_size)
parser.add_argument("--epochs", action = "store", type = int, default = epochs)
parser.add_argument("--batch_size", action = "store", type = int, default = batch_size)
parser.add_argument("--learning_rate", action = "store", type = float, default = learning_rate)
parser.add_argument("--seed", action = "store", default = 42)
parser.add_argument("--name", action = "store", default = None, type = str)
parser.add_argument("--notebook", action = "store", default = False, type = bool)
input = parser.parse_args(sys.argv[1:])




################################################
# Get Input from terminal                      # 
################################################
Model, Data   = input.Model, input.Data
learning_rate = input.learning_rate
batch_size    = input.batch_size
epochs        = input.epochs
train_size    = input.train_size
seed          = input.seed

if input.notebook:
    print("Notebook style loading bars")
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if Model[-3:] != ".py":
    Model += ".py"
if Data[-3:] != ".py":
    Data += ".py"

# Check if the model is valid
if Model[:-3] in os.listdir(osp.join(file_path, "models", "saved_models")):
    model = load_model(osp.join(file_path, "models", "saved_models", Model[:-3]))
    print("Loaded existing model")
    loaded_model = True
elif Model not in os.listdir(osp.join(file_path, "models")):
    sys.exit(f"Model file {Model} not found in models/")
else:
    model = None
    loaded_model = False

if Data not in os.listdir(osp.join(file_path, "data")):
    sys.exit(f"Data file {Data} not found in data/")



################################################
# Load Data and Model                          # 
################################################
sys.path.append(osp.join(file_path, "data"))
data = getattr(importlib.import_module(Data[:-3]), Data[:-3])()

if not loaded_model:
    sys.path.append(osp.join(file_path, "models"))
    model = getattr(importlib.import_module(Model[:-3]), "model")()
    print("Loaded class")


model.compile()
print("Model and data found succesfully")

# Setup save path
if input.name:
    save_path = osp.join(file_path, "models", "saved_models", input.name)
else:
    save_path = osp.join(file_path, "models", "saved_models", Model[:-3])

if not os.path.exists(save_path):
    os.mkdir(save_path)
################################################
# Split Data                                   # 
################################################

# Setup train-test split and loader for the data
from spektral.data import DisjointLoader

np.random.seed(seed)

idxs = np.random.permutation(len(data))
split = int(train_size * len(data))
idx_tr, idx_test  = np.split(idxs, [split])
dataset_train, dataset_test = data[idx_tr], data[idx_test]

loader_train  = DisjointLoader(dataset_train, epochs = epochs, batch_size = batch_size)
loader_test   = DisjointLoader(dataset_test, batch_size = batch_size, epochs = 1)



################################################
# Define Loss and Optimzier                    # 
################################################

################################################
# Loss and Optimation                          # 
################################################

opt           = Adam(learning_rate = learning_rate)
mse           = MeanSquaredError()

def loss_func(y_reco, y_true):
    # Energy loss
    loss      = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    # Position loss
    loss     += tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    )

    loss      += tf.reduce_mean(
        tf.math.acos(tf.reduce_sum(y_reco[:, 4:] * y_true[:, 4:], axis = 1) /
        tf.sqrt(tf.reduce_sum(y_reco[:, 4:] ** 2, axis = 1) * tf.sqrt(tf.reduce_sum(y_true[:, 4:] ** 2, axis = 1))))
        )

    # loss      += tf.reduce_mean(tf.abs(1 - tf.reduce_sum(y_reco[:, 4:] ** 2 , axis = 1)))

    # loss    += mse(y_reco[:, 4:], y_true[:, 4:])
    return loss

def loss_func_from(y_reco, y_true):
    # Energy loss
    loss_energy = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    # Position loss
    loss_dist  = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    )
    # Angle loss
    loss_angle = tf.reduce_mean(
        tf.math.acos(tf.reduce_sum(y_reco[:, 4:] * y_true[:, 4:], axis = 1) /
        tf.sqrt(tf.reduce_sum(y_reco[:, 4:] ** 2, axis = 1) * tf.sqrt(tf.reduce_sum(y_true[:, 4:] ** 2, axis = 1))))
        )
    # loss_angle += tf.reduce_mean(tf.abs(1 - tf.reduce_sum(y_reco[:, 4:] ** 2 , axis = 1)))
    
    return float(loss_energy), float(loss_dist), float(loss_angle)

def metrics(y_reco, y_true):
    # Energy metric
    energy_residuals = y_true[:, 0] - y_reco[:, 0]
    energy_quantiles = tfp.stats.percentile(energy_residuals, [25, 75])
    w_energy         = (energy_quantiles[1] - energy_quantiles[0]) / 1.349


    # Distanc metric
    dist_resi  = tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    u_pos           = tfp.stats.percentile(dist_resi, [68])


    # Angle metric
    angle_resi = 180 / np.pi * tf.math.acos(tf.reduce_sum(y_reco[:, 4:] * y_true[:, 4:], axis = 1) /
        tf.sqrt(tf.reduce_sum(y_reco[:, 4:] ** 2, axis = 1) * tf.sqrt(tf.reduce_sum(y_true[:, 4:] ** 2, axis = 1))))
    
    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_pos.numpy()), float(u_angle.numpy())

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
        predictions = model(inputs, training = False)
        targets     = tf.cast(targets, tf.float32)
        loss        = loss_func(predictions, targets)
        loss       += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
def test_step(inputs, targets):
    predictions = model(inputs)
    targets     = tf.cast(targets, tf.float32) 
    out         = loss_func(predictions, targets)

    return predictions, targets, out


def validation(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000
        inputs[0][:, 3] = inputs[0][:, 3] / 10000
        targets[:, 1:4] = targets[:, 1:4] / 1000
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)
    
    y_reco  = tf.concat(prediction_list, axis = 0)
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32)

    w_energy, u_pos, u_angle = metrics(y_reco, y_true)
    l_energy, l_pos, l_angle = loss_func_from(y_reco, y_true)
    loss                     = loss_func(y_reco, y_true)

    return loss, [l_energy, l_pos, l_angle], [w_energy, u_pos, u_angle]


def test(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000
        inputs[0][:, 3] = inputs[0][:, 3] / 10000
        targets[:, 1:4] = targets[:, 1:4] / 1000
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)

    y_reco  = tf.concat(prediction_list, axis = 0).numpy()
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32).numpy()

    energy = y_true[:, 0]
    counts, bins = np.histogram(energy, bins = 10)

    xs = (bins[1:] + bins[: -1]) / 2

    w_energies, u_distances, u_angles = [], [], []

    for i in range(len(bins)-1):
        idx = np.logical_and(energy > bins[i], energy < bins[i + 1])

        w, u_dist, u_angle = metrics(y_true[idx, :], y_reco[idx, :])

        w_energies.append(w)
        u_distances.append(u_dist)
        u_angles.append(u_angle)


    fig, ax = plt.subplots(ncols = 3, figsize = (12, 4))

    # for a in ax:
    #     a.step(xs, counts, color = "gray", zorder = 10, alpha = 0.7)
    #     # a.yscale("log")
    #     a.set_xlabel("Log Energy")
    

    # Energy reconstruction
    ax[0].scatter(xs, w_energies)
    ax[0].set_title("Energy Performance")
    ax[0].set_ylabel(r"$w(\Deltalog(E)$")


    # Angle reconstruction
    ax[1].scatter(xs, u_angles)
    ax[1].set_title("Angle Performance")
    ax[1].set_ylabel(r"$u(\Delta\Omega)$")

    # Distance reconstruction
    ax[2].scatter(xs, u_distances)
    ax[2].set_title("Distance Performance")
    ax[2].set_ylabel(r"$u(||y_{reco} - y_{true}||)$")

    return(fig, ax)
