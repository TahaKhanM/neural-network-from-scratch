# Neural network from scratch

A NumPy implementation of dense layers, backpropagation and mini-batch stochastic gradient descent. It classifies handwritten MNIST digits without an automatic differentiation library. The forward pass, loss and gradients are all in [main.py](main.py).

A recorded ten-epoch run of the `784 → 100 → 20 → 10` network reached **96.18% accuracy on the 10,000 official test images**. [Results and hyperparameters](results/mnist-seed7.json).

## Run it

Python 3.11+ and NumPy are the only runtime requirements. The four MNIST IDX files are already included in `Data/`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python main.py --epochs 10 --metrics results/my-run.json
```

For a short smoke run:

```bash
python main.py --epochs 1 --train-limit 500 --validation-size 500 --hidden 16 --batch-size 64
```

The CLI accepts `--seed`, `--hidden`, `--learning-rate`, `--batch-size` and `--data`; run `python main.py --help` for details. Importing `main` does not load data or start training. `Network([2, 3, 2], seed=7)` also works independently of MNIST.

## The implementation

All executable logic is in [main.py](main.py). Inputs use columns for examples: a batch has shape `(features, cases)`. A layer stores weights `(outputs, inputs)` and a bias `(outputs, 1)`, so the forward operation is `sigmoid(W @ A + b)` without transposing parameter arrays.

The loss is the mean squared error over **both output neurons and examples**:

$$L = \frac{1}{Km}\sum_{k=1}^{K}\sum_{j=1}^{m}(A_{kj}-Y_{kj})^2.$$

Backpropagation starts with $\delta_L=2(A_L-Y)\odot A_L\odot(1-A_L)/(Km)$. At each layer, `dW = delta @ previous_activation.T` and `db = delta.sum(axis=1, keepdims=True)`. Propagating through `W.T` and the previous sigmoid derivative yields the next delta. The division happens once, matching the reported loss exactly. All gradients are computed before any parameter is updated.

Sigmoid uses `exp(-abs(z))` with separate positive/negative expressions to avoid overflow without clipping the mathematical function. Weight standard deviation is `1/sqrt(fan_in)` and biases start at zero; this avoids the large initial pre-activations of unscaled Gaussian weights. Training shuffles the cases each epoch using an explicit random generator. A short final batch is included and epoch losses are weighted by its actual size.

The network keeps sigmoid outputs and MSE to retain the original project's derivation. This costs learning efficiency: sigmoid saturates and MSE adds another small derivative at the output. Softmax with cross-entropy would be a sensible alternative for mutually exclusive classes. The present outputs are independent scores in `[0,1]`, **not a calibrated probability distribution**; predictions use their argmax. Training and evaluation use the same forward computation.

## Evaluation and evidence

The reported run uses seed 7, learning rate 10, batch size 100 and ten fixed epochs. A seeded shuffle reserves 5,000 examples from the official 60,000-image training set for validation, leaving 55,000 for fitting. The official test set is evaluated once after training. The resulting validation accuracy is **95.90%**, test accuracy **96.18%** and test MSE **0.006486**. This is one run, with no claim of uncertainty across random seeds or a hyperparameter search. Test results should not be used to choose subsequent hyperparameters.

`online_train_mse` is an average of losses measured before each mini-batch update, across changing parameters. Validation and test losses use one fixed network. Their meanings differ, so the training number should not be read as an exact end-of-epoch training-set loss.

The six tests check:

- Every weight and bias gradient against central finite differences on a small multilayer, multicase network.
- Learning a small classification problem, including a partial batch and batch size larger than the dataset.
- Stable sigmoid evaluation at extreme inputs, deterministic training, input/label validation and single-case shapes.
- IDX header and exact payload-length validation, including truncated data.

CI runs these tests and a small MNIST training/evaluation run. The full result was produced with Python 3.13 and NumPy 2.5.3; floating-point rounding and BLAS implementations can cause small differences on another machine. For similar CPU thread settings, prefix the command with `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

## Background and limits

The [original notebook](historical/original-mnist.ipynb) preserves the derivation and acknowledges Michael Nielsen's *Neural Networks and Deep Learning*. MNIST is by Yann LeCun, Corinna Cortes and Christopher Burges.

The later implementation fixes loss and gradient scaling, removes import-time training and adds explicit seeds and data partitions. The network keeps sigmoid and MSE so the derivation stays easy to follow. Softmax with cross-entropy would be a useful next comparison.

This is a small educational model. It has no convolution, regularisation, checkpointing or GPU path. The reported accuracy comes from one fixed run rather than a search across architectures or seeds.
