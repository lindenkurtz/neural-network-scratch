# Neural Network from Scratch (NumPy, CIFAR-10)

A multilayer perceptron trained on the CIFAR-10 dataset, implemented in Python with only NumPy. This is an early-summer warm-up project — my goal was to ground myself in the math and code behind basic neural networks before moving on to more advanced work.

## My process

I wanted to focus on the underlying mathematics and the translation into code, so I derived all the math by hand and implemented backprop without referencing existing implementations or coding tutorials.

The main resource I used was 3Blue1Brown's series on neural networks, which walks through how linear algebra, calculus, and statistics combine to classify images (using MNIST as the example dataset).

## What's implemented

- Recursive forward pass through arbitrary fully-connected architectures
- Backpropagation derived and implemented by hand, including the softmax + cross-entropy gradient shortcut (`∂L/∂z = ŷ − y`)
- Numerically stable sigmoid and softmax (max-subtraction in softmax; the `np.where` trick for sigmoid to avoid overflow on large negative inputs)
- Xavier/He-style weight initialization
- Mini-batch stochastic gradient descent with gradient clipping
- Cross-entropy loss

## Architecture

`3072 (32×32×3 flattened) → 128 → 64 → 32 → 10`

Sigmoid activations in hidden layers, softmax on the output.

## Setup

1. Download the CIFAR-10 Python version from https://www.cs.toronto.edu/~kriz/cifar.html
2. Extract it so that `cifar-10-batches-py/` sits next to the script
3. Install dependencies: `pip install numpy tqdm`

## Usage

```bash
python main.py
```

Training runs for 50 epochs with batch size 32 on the first data batch. The script prints the cost every 10 epochs.

## Results

*(To be filled in once training completes — final cross-entropy loss, accuracy on a held-out batch, and a loss curve.)*

## What I struggled with

The calculus came together quickly for single neurons and single inputs, but extending it to vectorized layers and batched inputs took real work. A lot of the project was sitting with the dimensions and figuring out whether a given step needed matrix multiplication, element-wise multiplication, or a sum along a particular axis — and which order of operations would produce the shapes I wanted.

## Limitations and next steps

- **Redundant computation in backprop.** My backprop currently re-runs `forward()` at every layer to recompute activations, rather than caching them on the forward pass. This is wasteful and would be a real bottleneck on larger networks.
- **No modern optimizer or regularization.** Vanilla SGD with gradient clipping only — no Adam, no momentum, no dropout, no batch norm.
- **Sigmoid hidden activations.** ReLU would likely train faster and reach better accuracy.
- **Only trained on one data batch.** Should iterate over all five training batches and evaluate on the test batch.

Future work for this codebase, if I return to it, would be caching activations on the forward pass, swapping in ReLU and Adam, and adding a proper train/test split.