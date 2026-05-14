import numpy as np
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

#generator
rng = np.random.default_rng(seed=5)

#---------------------Data Wrangling----------------------

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle('cifar-10-batches-py/data_batch_1')
images = batch1[b'data']
labels = batch1[b'labels']

#---------------------Model Definition----------------------

def sigmoid(x):
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

def inv_sigmoid(x):
    return np.log(x / (1 - x))

def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()

def cost_function(pred_y, real_y):
    return - np.sum(real_y * np.log(pred_y))
        

class Model:
    def __init__(self, LAYER_SIZES=np.array([]), LR=0.0001):
        self.NUM_LAYERS = LAYER_SIZES.size - 1
        self.LR = LR
        
        # Biases
        self.biases=[]
        for size in LAYER_SIZES:
            if (size == LAYER_SIZES[0]):
                continue
            self.biases.append(rng.standard_normal(size) * np.sqrt(2.0 / size))

        # Weights
        self.weights=[]
        for i in range(LAYER_SIZES.size):
            if (LAYER_SIZES[i] == LAYER_SIZES[-1]):
                break
            self.weights.append(rng.standard_normal((LAYER_SIZES[i], LAYER_SIZES[i+1])) * np.sqrt(1.0 / LAYER_SIZES[i])) # use std norm so weights can be negative, and uses Xavier to make smaller


    def forward(self, input_arr, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if layer_idx == 0:
            return input_arr
        if layer_idx == self.NUM_LAYERS:
            return softmax(self.forward(input_arr, layer_idx - 1) @ self.weights[layer_idx - 1] + self.biases[layer_idx - 1])
        return sigmoid(self.forward(input_arr, layer_idx - 1) @ self.weights[layer_idx - 1] + self.biases[layer_idx - 1])

    
    def backprop_helper(self, dc_da, input_arr, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if (layer_idx == 0):
            return
        
        dc_da_1 = (dc_da * self.forward(input_arr, layer_idx) * (1 - self.forward(input_arr, layer_idx))) @ self.weights[layer_idx - 1].T

        self.backprop_helper(dc_da_1, input_arr, layer_idx - 1)
        if (layer_idx == self.NUM_LAYERS):
            dc_dw = self.forward(input_arr, layer_idx - 1).T @ dc_da
            dc_db = np.mean(dc_da, axis=0)
        else:
            dc_dw = self.forward(input_arr, layer_idx - 1).T @ (self.forward(input_arr, layer_idx) * (1 - self.forward(input_arr, layer_idx)) * dc_da) 
            dc_db = np.mean(self.forward(input_arr, layer_idx) * (1 - self.forward(input_arr, layer_idx)) * dc_da, axis=0)
        
        dc_dw = np.clip(dc_dw, -1, 1) # gradient clipping
        
        self.weights[layer_idx - 1] -= dc_dw * self.LR # -= instead of +=
        self.biases[layer_idx - 1] -= dc_db * self.LR

        return
    
    def backprop(self, input_arr, target_arr):
        output_arr = self.forward(input_arr)
        dc_dz = output_arr - target_arr
        
        self.backprop_helper(dc_dz, input_arr)


    
#---------------------Running Model----------------------

model = Model(np.array([images[0].size, 16, 16, 16, 10]))

print(labels[0])
print(model.forward(images[0]))

labels_onehot = np.eye(10)[labels]

for epoch in tqdm(range(25)):
    #shuffle order of inputs and outputs
    p = np.random.permutation(len(images))
    images = images[p]
    labels_onehot = labels_onehot[p]
    if epoch % 5 == 0:
        print(cost_function(model.forward(images), labels_onehot))

    model.backprop(images, labels_onehot)

print(np.max(model.weights[0].flatten()))
print(labels_onehot[0])
print(model.forward(images[0]))
print(cost_function(model.forward(images), labels_onehot))