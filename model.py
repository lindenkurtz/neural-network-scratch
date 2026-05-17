import numpy as np
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

#generator
rng = np.random.default_rng(seed=5)

#---------------------Data Wrangling----------------------
#get dictionary of cifar-10 data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle('cifar-10-batches-py/data_batch_1')
images = batch1[b'data'] / 255.0 #normalizes
labels = batch1[b'labels']

#images = images[:10] # for testing
#labels = labels[:10]

#---------------------Model Definition----------------------

def sigmoid(x): # 1 / (1 + exp(-x))
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

def inv_sigmoid(x): # ln(x / (1 - x))
    return np.log(x / (1 - x))

def softmax(x): # each element x_i in a vector x = exp(x_i) / sum(x)
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)

def cost_function(pred_y, real_y): # y * ln(ŷ)
    return - np.sum(real_y * np.log(pred_y + 1e-10)) # add a small number so I don't take log of 0
        

class Model:
    def __init__(self, LAYER_SIZES=np.array([]), LR=0.002):
        self.NUM_LAYERS = LAYER_SIZES.size - 1
        self.LR = LR
        
        # Biases - biases[0] corresponds with the first hidden layer
        self.biases=[]
        for size in LAYER_SIZES:
            if (size == LAYER_SIZES[0]):
                continue
            self.biases.append(rng.standard_normal(size) * np.sqrt(2.0 / size))

        # Weights - weights[0] corresponds with the matrix connecting the input layer to the first hidden layer
        self.weights=[]
        for i in range(LAYER_SIZES.size):
            if (LAYER_SIZES[i] == LAYER_SIZES[-1]):
                break
            self.weights.append(rng.standard_normal((LAYER_SIZES[i], LAYER_SIZES[i+1])) * np.sqrt(1.0 / LAYER_SIZES[i])) # use std norm so weights can be negative, and uses Xavier to make smaller


    def forward(self, input_arr, layer_idx=None):
        """
        Uses recursion to calculate an output from the neural network's input, weights, and biases

        Args:
            input_arr (int vec or matrix): a vector representing the state of an input,
                                           or a matrix combining batches of these vectors for different states.

        Returns:
            A vector or matrix represting either a single list of probabilites of each output,
            or a batch of these lists, respectively
        """
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if layer_idx == 0:
            return input_arr
        if layer_idx == self.NUM_LAYERS:
            return softmax(self.forward(input_arr, layer_idx - 1) @ self.weights[layer_idx - 1] + self.biases[layer_idx - 1])
        return sigmoid(self.forward(input_arr, layer_idx - 1) @ self.weights[layer_idx - 1] + self.biases[layer_idx - 1]) #only for last layer

    
    def backprop_helper(self, dc_da, input_arr, layer_idx=None):
        """
        Uses recursion to find the errors in the network that caused differences between the predicted and
        actual output, and modifies network weights and biases to minimize this.

        Args:
            dc_da (double): on the first pass / final layer, represents the derivative of weighted sum of the output layer with respect to the cost function
                            otherwise, represents the derivative of the activation of the current layer of the network with respect to the cost
            input_arr (int matrix): a matrix of inputs by batches, where inputs represent the state of an input for different states.

        Outputs:
            None, but updates the network for an object.
        """
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if (layer_idx == 0):
            return
        
        if (layer_idx == self.NUM_LAYERS):
            dc_da_1 = dc_da @ self.weights[layer_idx - 1].T
        else:
            dsig = self.forward(input_arr, layer_idx) * (1 - self.forward(input_arr, layer_idx))
            dc_da_1 = (dc_da * dsig) @ self.weights[layer_idx - 1].T

        self.backprop_helper(dc_da_1, input_arr, layer_idx - 1)
        if (layer_idx == self.NUM_LAYERS):
            dc_dw = self.forward(input_arr, layer_idx - 1).T @ dc_da
            dc_db = np.sum(dc_da, axis=0) # sum and not mean because matrix multiplication for the weights acts like a summation
        else:
            dc_dw = self.forward(input_arr, layer_idx - 1).T @ (dsig * dc_da) 
            dc_db = np.sum(dsig * dc_da, axis=0)
        
        dc_dw = np.clip(dc_dw, -1, 1) # gradient clipping
        
        self.weights[layer_idx - 1] -= dc_dw * self.LR # -= instead of +=, as the gradient shows the direction of fastest increase.
        self.biases[layer_idx - 1] -= dc_db * self.LR

        return
    
    def backprop(self, input_arr, target_arr):
        """
        Updates model parameters based on the cost function. See backprop_helper
        """
        output_arr = self.forward(input_arr)
        dc_dz = output_arr - target_arr
        
        self.backprop_helper(dc_dz, input_arr)


    
#---------------------Running Model----------------------

model = Model(np.array([images[0].size, 128, 64, 32, 10])) # widen layers to preserve information
BATCH_SIZE = 32

print(labels[0])
print(model.forward(images[0]))

labels_onehot = np.eye(10)[labels]

for epoch in tqdm(range(50), desc="Epoch"):
    #shuffle order of inputs and outputs
    p = np.random.permutation(len(images))
    images = images[p]
    labels_onehot = labels_onehot[p]
    if epoch % 10 == 0:
        print(cost_function(model.forward(images), labels_onehot))

    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Batches", leave=False):
        img_batch = images[i : i + BATCH_SIZE]
        lbl_batch = labels_onehot[i : i + BATCH_SIZE]
        model.backprop(img_batch, lbl_batch)

print(np.max(model.weights[0].flatten()))
print(labels_onehot[0])
print(model.forward(images[0]))
print(cost_function(model.forward(images), labels_onehot))