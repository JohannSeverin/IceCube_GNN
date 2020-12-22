import os, sys, argparse, importlib, time

import numpy as np
import os.path as osp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('gpu') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
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

# ! Can be generalized more !

opt           = Adam(learning_rate = learning_rate)
loss_func     = MeanSquaredError()



################################################
# Train Model                                  # 
################################################

# Define Train Step
@tf.function(input_signature = loader_train.tf_signature(), experimental_relax_shapes = True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss        = loss_func(predictions, target)
        loss       += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

print("Fitting model")
current_batch = 0
loss          = 0
current_epoch = 0
epoch_start   = time.time()

for batch in loader_train:
    inputs, target = batch
    target = target.reshape(-1, 1)
    out    = train_step(inputs, target)
    loss  += out
    current_batch += 1 
    print(f"Completed: \t {current_batch} \t / {loader_train.steps_per_epoch} \t current_loss: {out:.4f}", end ='\r' )
    sys.stdout.flush()
    if current_batch == loader_train.steps_per_epoch:
        current_epoch += 1
        print(f"Loss after epoch {current_epoch} of {epochs}: {loss / loader_train.steps_per_epoch:.4f} \t in {time.time() - epoch_start:.1f} seconds")
        epoch_start = time.time()
        loss = 0
        current_batch = 0

print("Fitting done")

################################################
# Test Model                                   # 
################################################

print("Testing model")

loss = 0
current_batch = 0 
for batch in loader_test:
    inputs, target = batch
    target = target.reshape(-1, 1)
    predictions = model(inputs)
    out    = loss_func(predictions, target)
    loss  += out
    current_batch += 1
    print(f"completed: \t {current_batch} \t / {loader_test.steps_per_epoch} \t current_loss: {out}", end ='' )

    
print(f" \n Done, test loss:{loss / loader_test.steps_per_epoch:.3f}")


if input.name:
    save_path = osp.join(file_path, "models", "saved_models", input.name)
else:
    save_path = osp.join(file_path, "models", "saved_models", Model[:-3])

if not os.path.exists(save_path):
    os.mkdir(save_path)

model.save(save_path)
print("Model saved")



