import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

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
    return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x / (1 - x))

def cost_function(pred_y, real_y):
    return np.mean(np.sum(np.square(real_y - pred_y)))
        

class Model:
    def __init__(self, LAYER_SIZES=np.array([]), LR=0.001):
        self.NUM_LAYERS = LAYER_SIZES.size - 1
        self.LR = LR
        
        # Biases
        self.biases=[]
        for size in LAYER_SIZES:
            if (size == LAYER_SIZES[0]):
                continue
            self.biases.append(np.zeros(size))

        # Weights
        self.weights=[]
        for i in range(LAYER_SIZES.size):
            if (LAYER_SIZES[i] == LAYER_SIZES[-1]):
                break
            self.weights.append(np.zeros((LAYER_SIZES[i], LAYER_SIZES[i+1])))


    def forward(self, input_arr, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if layer_idx == 0:
            return input_arr
        return sigmoid(self.forward(input_arr, layer_idx - 1) @ self.weights[layer_idx - 1] + self.biases[layer_idx - 1])

    
    def backprop_helper(self, dc_da, input_arr, layer_idx=None):
        if layer_idx is None:
            layer_idx = self.NUM_LAYERS
        
        if (layer_idx == 0):
            return
        
        #figure out correct matrix multiplication for all below

        dc_da_1 = (dc_da * self.forward(input_arr, layer_idx) * (1 - self.forward(input_arr, layer_idx))) @ self.weights[layer_idx - 1].T

        self.backprop_helper(dc_da_1, input_arr, layer_idx - 1)

        dc_dw = self.forward(input_arr, layer_idx - 1).T @ (self.forward(input_arr, layer_idx) * 1 - self.forward(input_arr, layer_idx) * dc_da)
        
        dc_db = self.forward(input_arr, layer_idx) * 1 - self.forward(input_arr, layer_idx) * dc_da
        
        self.weights[layer_idx - 1] += dc_dw * self.LR
        self.biases[layer_idx - 1] += dc_db * self.LR

        return
    
    def backprop(self, input_arr, target_arr):
        output_arr = self.forward(input_arr)
        dc_da = np.mean(np.sum(2*(output_arr - target_arr), axis=0))
        
        self.backprop_helper(dc_da, input_arr)


    
#---------------------Running Model----------------------

model = Model(np.array([images[0].size, 16, 16, 16, 10]))

print(labels[0])
print(model.forward(images[0]))

labels_onehot = np.eye(10)[labels]
model.backprop(images, labels_onehot)

print(labels[0])
print(model.forward(images[0]))

