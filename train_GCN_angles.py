import os, sys, argparse, importlib, time
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from tqdm import tqdm

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

file_path = osp.dirname(osp.realpath(__file__))

################################################
# Setup Deafult Variabls                       # 
################################################
learning_rate = 2.5e-4
batch_size    = 512
epochs        = 100
early_stop    = True
patience      = 3
model_name    = "None"



################################################
# Get model and data                           # 
################################################
from models.GCN_angles import model
model = model()
# model = load_model(osp.join(file_path, "models", "saved_models", "MessPass1"))



from data.graph_w_edge2_angles import graph_w_edge2
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

    # azimuthal = y_reco[:, 4]
    # zenith    = y_reco[:, 5]
    loss     += tf.reduce_mean(
        1 - (tf.cos(y_reco[:, 5]) * tf.cos(y_true[:, 5]) + tf.sin(y_reco[:, 5]) * tf.sin(y_true[:, 5]) * tf.cos(y_reco[:, 4] - y_true[:, 4])))

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
    loss_angle  = tf.reduce_mean(
       1 - (tf.cos(y_reco[:, 5]) * tf.cos(y_true[:, 5]) + tf.sin(y_reco[:, 5]) * tf.sin(y_true[:, 5]) * tf.cos(y_reco[:, 4] - y_true[:, 4])))
    # loss_angle = tf.reduce_mean(
    #     tf.math.acos(tf.reduce_sum(y_reco[:, 4:] * y_true[:, 4:], axis = 1) /
    #     tf.sqrt(tf.reduce_sum(y_reco[:, 4:] ** 2, axis = 1) * tf.sqrt(tf.reduce_sum(y_true[:, 4:] ** 2, axis = 1))))
    #     )
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
    angle_resi = 180 / np.pi * tf.math.acos(tf.cos(y_reco[:, 5]) * tf.cos(y_true[:, 5]) + tf.sin(y_reco[:, 5]) * tf.sin(y_true[:, 5]) * tf.cos(y_reco[:, 4] - y_true[:, 4]))
    
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
        predictions = model(inputs, training = True)
        targets     = tf.cast(targets, tf.float32)
        loss        = loss_func(predictions, targets)
        loss       += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



@tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
def test_step(inputs, targets):
    predictions = model(inputs, training = False)
    targets     = tf.cast(targets, tf.float32) 
    out         = loss_func(predictions, targets)

    return predictions, targets, out


def validation(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        inputs[0][:, :3] = inputs[0][:, :3] / 1000
        inputs[0][:, 3] = inputs[0][:, 3]
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

    for a in ax:
        a_ = a.twinx()
        a_.step(xs, counts, color = "gray", zorder = 10, alpha = 0.7)
        a_.set_yscale("log")
        a.set_xlabel("Log Energy")
    

    # Energy reconstruction
    ax[0].scatter(xs, w_energies)
    ax[0].set_title("Energy Performance")
    ax[0].set_ylabel(r"$w(\Delta log(E)$")


    # Angle reconstruction
    ax[1].scatter(xs, u_angles)
    ax[1].set_title("Angle Performance")
    ax[1].set_ylabel(r"$u(\Delta \Omega)$")

    # Distance reconstruction
    ax[2].scatter(xs, u_distances)
    ax[2].set_title("Distance Performance")
    ax[2].set_ylabel(r"$u(||y_{reco} - y_{true}||)$")

    fig.tight_layout()

    return fig, ax


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
learning_Rate = next(lr_gen)


for batch in loader_train:
    inputs, targets = batch
    inputs[0][:, :3] = inputs[0][:, :3] / 1000
    inputs[0][:, 3] = inputs[0][:, 3]
    targets[:, 1:4] = targets[:, 1:4] / 1000
    out             = train_step(inputs, targets)
    loss           += out

    current_batch  += 1
    pbar.update(1)
    pbar.set_description(f"Epoch {current_epoch} / {epochs}; Avg_loss: {loss / current_batch:.6f}")


    if current_batch == loader_train.steps_per_epoch:
        
        print(f"Epoch {current_epoch} of {epochs} done in {time.time() - start_time:.2f} seconds using learning rate: {learning_rate:.2E}")
        print(f"Avg loss of train: {loss / loader_train.steps_per_epoch:.6f}")

        loader_val    = DisjointLoader(dataset_val, epochs = 1,      batch_size = batch_size)
        val_loss, val_loss_from, val_metric = validation(loader_val)

        print(f"Avg loss of validation: {val_loss:.6f}")
        print(f"Loss from:  Energy: {val_loss_from[0]:.6f} \t Position: {val_loss_from[1]:.6f} \t Angle: {val_loss_from[2]:.6f} ")
        print(f"Energy: w = {val_metric[0]:.6f} \t Position: u = {val_metric[1]:.6f} \t Angle: u = {val_metric[2]:.6f}")

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

        
fig, ax = test(loader_test)
fig.savefig(f"model_tests/{model_name}_test.pdf")