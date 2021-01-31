import os, sys, argparse, importlib, time
import numpy as np
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
learning_rate = 3e-4
batch_size    = 512
epochs        = 10




################################################
# Get model and data                           # 
################################################
from models.GCN import model
model = model()


from data.graph_w_edge2 import graph_w_edge2
dataset = graph_w_edge2()
idx_lists = dataset.index_lists


# Split data
dataset_train = dataset[idx_lists[0]]
dataset_val   = dataset[idx_lists[1]]
dataset_test  = dataset[idx_lists[2]]

# Make loaders
loader_train  = DisjointLoader(dataset_train, epochs = epochs, batch_size = batch_size)
loader_test   = DisjointLoader(dataset_test , epochs = 1,      batch_size = batch_size)




################################################
# Loss and Optimation                          # 
################################################

opt           = Adam(learning_rate = learning_rate)

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
    loss     += 1 / 1000 * tf.reduce_mean(
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

    loss     += tf.reduce_mean(
        tf.math.log(tf.math.cosh(tf.math.acos(
            tf.math.divide_no_nan(
                tf.reduce_sum(
                    tf.multiply(
                        y_reco[:, 4:], y_true[:, 4:]
                    ), axis = 1
                ),
                tf.multiply(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                y_reco[:, 4:]
                            ), axis = 1
                        )
                    ),
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                y_true[:, 4:]
                            ), axis = 1
                        )
                    )
                )
            )
        )))
    )
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
    loss_dist  = 1 / 1000 * tf.reduce_mean(
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
    # loss_angle = tf.reduce_mean(
    #     tf.math.acos(
    #         tf.sin(y_reco[:, 5]) * tf.sin(y_true[:, 5]) + tf.cos(y_reco[:, 5]) * tf.cos(y_true[:, 5]) * tf.cos(y_true[:,4] - y_reco[:, 4]) 
    #     ))
    loss_angle = tf.reduce_mean(
        tf.math.log(tf.math.cosh(tf.math.acos(
            tf.math.divide_no_nan(
                tf.reduce_sum(
                    tf.multiply(
                        y_reco[:, 4:], y_true[:, 4:]
                    ), axis = 1
                ),
                tf.multiply(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                y_reco[:, 4:]
                            ), axis = 1
                        )
                    ),
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                y_true[:, 4:]
                            ), axis = 1
                        )
                    )
                )
            )
        )))
    )
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
    
    # angle_resi = 180 / np.pi * tf.math.acos(
    #         tf.sin(y_reco[:, 5]) * tf.sin(y_true[:, 5]) + tf.cos(y_reco[:, 5]) * tf.cos(y_true[:, 5]) * tf.cos(y_true[:,4] - y_reco[:, 4]) 
    #     )

    angle_resi = 180 / np.pi * tf.math.acos(
        tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.multiply(
                    y_reco[:, 4:], y_true[:, 4:]
                ), axis = 1
            ),
            tf.multiply(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            y_reco[:, 4:]
                        ), axis = 1
                    )
                ),
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            y_true[:, 4:]
                        ), axis = 1
                    )
                )
            )
        )
    )
    u_angle         = tfp.stats.percentile(angle_resi, [68])

    return float(w_energy.numpy()), float(u_pos.numpy()), float(u_angle.numpy())


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






################################################
# Training                                     # 
################################################

current_batch = 0
current_epoch = 1
loss          = 0



pbar          = tqdm(total = loader_train.steps_per_epoch, position = 0, leave = True)
start_time    = time.time()

for batch in loader_train:
    inputs, targets = batch
    out             = train_step(inputs, targets)
    loss           += out

    current_batch  += 1
    pbar.update(1)
    pbar.set_description(f"Epoch {current_epoch} / {epochs}; Avg_loss: {loss / current_batch:.6f}")


    if current_batch == loader_train.steps_per_epoch:
        
        print(f"Epoch {current_epoch} of {epochs} done in {time.time() - start_time:.2f} seconds")
        print(f"Avg loss of train: {loss / loader_train.steps_per_epoch:.6f}")

        loader_val    = DisjointLoader(dataset_val, epochs = 1,      batch_size = batch_size)
        val_loss, val_loss_from, val_metric = validation(loader_val)

        print(f"Avg loss of validation: {val_loss:.6f}")
        print(f"Loss from:  Energy: {val_loss_from[0]:.6f} \t Position: {val_loss_from[1]:.6f} \t Angle: {val_loss_from[2]:.6f} ")
        print(f"Energy: w = {val_metric[0]:.6f} \t Position: u = {val_metric[1]:.6f} \t Angle: u = {val_metric[2]:.6f}")
        
        if current_epoch != epochs:
            pbar          = tqdm(total = loader_train.steps_per_epoch, position = 0, leave = True)

        loss            = 0
        start_time      = time.time()
        current_epoch  += 1
        current_batch   = 0

