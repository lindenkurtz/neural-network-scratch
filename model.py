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

class Model:
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, HIDDEN_SIZE3, OUTPUT_SIZE):
        # Biases
        self.b1 = np.zeros(HIDDEN_SIZE1)
        self.b2 = np.zeros(HIDDEN_SIZE2)
        self.b3 = np.zeros(HIDDEN_SIZE2)

        # Weights
        self.w1 = np.zeros((INPUT_SIZE, HIDDEN_SIZE1))
        self.w2 = np.zeros((HIDDEN_SIZE1, HIDDEN_SIZE2))
        self.w3 = np.zeros((HIDDEN_SIZE2, HIDDEN_SIZE3))
        self.w4 = np.zeros((HIDDEN_SIZE3, OUTPUT_SIZE))

    def forward(self, input_arr):
        h1_arr = sigmoid(input_arr @ self.w1 + self.b1)
        h2_arr = sigmoid(h1_arr @ self.w2 + self.b2)
        h3_arr = sigmoid(h2_arr @ self.w3 + self.b3)
        output_arr = sigmoid(h3_arr @ self.w4)
        return output_arr
    
#---------------------Running Model----------------------

print(labels[0])
model = Model(images[0].size, 16, 16, 16, 10)

print(model.forward(images[0]))
