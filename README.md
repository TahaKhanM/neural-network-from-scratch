# Neural Network From Scratch (NumPy): MNIST

This project is a **from-scratch** implementation of a fully connected (dense) neural network trained on **MNIST** digit classification (0–9). It uses only **NumPy** for the actual neural network computations (plus `idx2numpy` to read MNIST IDX files and `matplotlib` for visualisation).

The entire project lives in a single file: **`main.py`**.

---

## How to run

Install dependencies:

```bash
pip install numpy idx2numpy matplotlib
```

Place the MNIST IDX files under `./Data` using the same paths referenced in `main.py`:

- `./Data/train-images.idx3-ubyte`
- `./Data/train-labels.idx1-ubyte`
- `./Data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte`
- `./Data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte`

Run:

```bash
python main.py
```

The script trains for a fixed number of epochs, prints the average mini-batch cost per epoch, then evaluates on the test set.

---

## The maths behind the network

### Shapes and notation (matches the code)

The code uses **column-major batches**:

- A mini-batch of inputs is a matrix

$$
X \in \mathbb{R}^{n_0 \times m}
$$

where:
- $n_0 = 784$ for MNIST ($28\times 28$ pixels flattened)
- $m$ is the number of examples in the batch

For each layer $\ell \in \{1,\dots,L\}$:

- Weights:

$$
W^{(\ell)} \in \mathbb{R}^{n_{\ell} \times n_{\ell-1}}
$$

- Biases:

$$
b^{(\ell)} \in \mathbb{R}^{n_{\ell} \times 1}
$$

- Pre-activations and activations:

$$
Z^{(\ell)} = W^{(\ell)}A^{(\ell-1)} + b^{(\ell)}, \qquad A^{(\ell)} = \sigma\big(Z^{(\ell)}\big)
$$

with $A^{(0)} = X$.

> **Important detail (implementation):** in `main.py`, biases are stored as row vectors of shape `(1, n_l)` but are used as column vectors via transpose (`b.T`) so they broadcast correctly across the batch.

---

### Activation: sigmoid

The network uses the sigmoid function at *every* layer:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Its derivative is:

$$
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)
$$

In backprop, this is used as an elementwise (Hadamard) factor.

---

### Cost function used in training (mean squared error)

Labels are converted to one-hot vectors (10 classes). For a single training example, the code computes a mean-squared style loss:

$$
C = \frac{1}{10}\sum_{k=1}^{10} \big(a_k^{(L)} - y_k\big)^2
$$

Over a mini-batch of size $m$, training effectively optimises the average cost.

---

### Backpropagation (what `backProp(...)` is doing)

Define the **error** at layer $\ell$ as:

$$
\delta^{(\ell)} = \frac{\partial C}{\partial Z^{(\ell)}}
$$

#### Output layer error

For MSE + sigmoid output, the chain rule gives:

$$
\delta^{(L)} = \frac{\partial C}{\partial A^{(L)}} \odot \sigma'\big(Z^{(L)}\big)
$$

and

$$
\frac{\partial C}{\partial A^{(L)}} \propto 2\big(A^{(L)} - Y\big)
$$

So the code computes (up to constant scaling):

$$
\delta^{(L)} = 2\big(A^{(L)} - Y\big) \odot \sigma'\big(Z^{(L)}\big)
$$

#### Propagating errors backwards

For earlier layers:

$$
\delta^{(\ell)} = \big(W^{(\ell+1)}\big)^T\delta^{(\ell+1)} \odot \sigma'\big(Z^{(\ell)}\big)
$$

This is exactly the pattern inside the loop that walks backwards through `layers`.

#### Gradients for weights and biases

For a mini-batch:

$$
\frac{\partial C}{\partial W^{(\ell)}} = \frac{1}{m}\,\delta^{(\ell)}\big(A^{(\ell-1)}\big)^T
$$

$$
\frac{\partial C}{\partial b^{(\ell)}} = \frac{1}{m}\sum_{i=1}^{m} \delta^{(\ell)}_{:,i}
$$

**Implementation note:** the code forms weight gradients as

$$
(A^{(\ell-1)})\,(\delta^{(\ell)})^T
$$

(which is the transpose of the conventional formula) and then transposes again during the parameter update.

---

### SGD parameter update

For learning rate $\eta$:

$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta\,\frac{\partial C}{\partial W^{(\ell)}}
$$

$$
b^{(\ell)} \leftarrow b^{(\ell)} - \eta\,\frac{\partial C}{\partial b^{(\ell)}}
$$

The function `stochasticGradientDescent(trainingRate)` loops over mini-batches and performs this update per batch.

---

## Where the maths appears in `main.py`

### Data prep

- MNIST images are normalised to $[0,1]$ by dividing by 255.
- Each image is flattened to a 784-vector.
- The dataset is transposed so that inputs have shape `(784, n_samples)`.

Functions:
- `createDesiredOutputs(labels)` builds the one-hot matrix $Y \in \mathbb{R}^{10 \times n}$.
- `displayImage(dataset, num)` renders an image column.

### Network representation

- `layers` is a list of layers.
- Each layer is stored as `[W, b]` where:
  - `W` has shape `(n_out, n_in)`
  - `b` has shape `(1, n_out)`

`initialiseNetwork([100, 20])` builds:

$$
784 \rightarrow 100 \rightarrow 20 \rightarrow 10
$$

### Forward pass

- `feedforward(inputs)` returns:
  - `finalOutput` = $A^{(L)}$
  - `weightedOutputs` = list of activations $A^{(1)},\dots,A^{(L)}$
  - `unweightedOutputs` = list of pre-activations $Z^{(1)},\dots,Z^{(L)}$

### Backward pass

- `backProp(...)` takes the stored $Z$ and $A$ values and returns averaged gradients for each layer.

### Batching

- `createBatches(data, labels, batchSize)` splits the matrices into equal-sized mini-batches.

### Test-time normalisation

- `feedforwardTest(inputs)` runs a forward pass and then applies `softmax` **column-wise**.
- Predictions are obtained via `argmax`.

---

## Necessary improvements implemented (minimal changes)

These changes keep your original structure intact (same data loading, same training objective), but fix correctness/numerical issues:

1. **Biases are now used in the forward pass**

   The network was initialising and updating biases, but not adding them during forward propagation. The forward pass now correctly computes:

   $$
   Z^{(\ell)} = W^{(\ell)}A^{(\ell-1)} + b^{(\ell)}
   $$

2. **Softmax is now a real (stable) softmax**

   Previously it was `x / sum(x)`. It is now:

   $$
   \mathrm{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\sum_j e^{z_j - \max(z)}}
   $$

3. **Batch splitting bug fixed**

   `np.array_split` requires an integer number of sections; the code previously passed a float. This is now cast to an `int`.

4. **Sigmoid made numerically safer**

   Inputs are clipped before `exp` to prevent overflow in extreme cases.


## Acknowledgements / references

The explanations and standard neural-network derivations in this README were informed by **Michael Nielsen’s free online book**:

- Michael A. Nielsen, *Neural Networks and Deep Learning* (Determination Press, 2015). Available at: `http://neuralnetworksanddeeplearning.com/`

