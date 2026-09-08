"""A dense sigmoid network and MNIST training loop implemented with NumPy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def sigmoid(z):
    # exp(-abs(z)) avoids overflow without changing the function by clipping.
    exp = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp), exp / (1 + exp))


def read_idx(path):
    """Read an unsigned-byte IDX tensor and reject malformed/truncated files."""
    raw = Path(path).read_bytes()
    if len(raw) < 4 or raw[:3] != b'\x00\x00\x08' or raw[3] == 0:
        raise ValueError(f'{path}: expected an unsigned-byte IDX file')
    dimensions = raw[3]
    header = 4 + 4 * dimensions
    if len(raw) < header:
        raise ValueError(f'{path}: truncated IDX header')
    shape = struct.unpack(f'>{dimensions}I', raw[4:header])
    if any(size == 0 for size in shape) or len(raw) - header != int(np.prod(shape)):
        raise ValueError(f'{path}: IDX shape does not match payload length')
    return np.frombuffer(raw, dtype=np.uint8, offset=header).reshape(shape)


def load_mnist(directory, split):
    prefix = 'train' if split == 'train' else 't10k'
    images = read_idx(directory / f'{prefix}-images.idx3-ubyte')
    labels = read_idx(directory / f'{prefix}-labels.idx1-ubyte')
    if images.ndim != 3 or images.shape[1:] != (28, 28) or labels.shape != (len(images),):
        raise ValueError('expected matching 28x28 MNIST images and labels')
    if np.any(labels > 9):
        raise ValueError('MNIST labels must be in 0..9')
    return images.reshape(len(images), -1).T.astype(np.float64) / 255, labels


def one_hot(labels, classes):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must be a one-dimensional integer array')
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError('label outside class range')
    return np.eye(classes)[labels].T


class Network:
    """Column batches: inputs (features, cases), weights (outputs, inputs)."""

    def __init__(self, sizes, seed=0):
        if len(sizes) < 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in sizes):
            raise ValueError('sizes must contain at least two positive integers')
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, 1 / np.sqrt(a), (b, a)) for a, b in zip(sizes, sizes[1:])]
        self.biases = [np.zeros((n, 1)) for n in sizes[1:]]

    def _inputs(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != self.sizes[0] or x.shape[1] == 0 or not np.all(np.isfinite(x)):
            raise ValueError('inputs must be finite, nonempty column batches with the expected feature count')
        return x

    def _forward(self, x):
        activations = [self._inputs(x)]
        for w, b in zip(self.weights, self.biases):
            activations.append(sigmoid(w @ activations[-1] + b))
        return activations

    def predict(self, x):
        return self._forward(x)[-1]

    def loss_and_gradients(self, x, y):
        activations = self._forward(x)
        output = activations[-1]
        y = np.asarray(y, dtype=float)
        if y.shape != output.shape or not np.all(np.isfinite(y)):
            raise ValueError('targets must be finite and match the output shape')
        difference = output - y
        loss = float(np.mean(difference ** 2))
        # Mean over BOTH classes and cases: the derivative matches the reported MSE.
        delta = 2 * difference * output * (1 - output) / output.size
        dw, db = [None] * len(self.weights), [None] * len(self.biases)
        for layer in reversed(range(len(self.weights))):
            dw[layer] = delta @ activations[layer].T
            db[layer] = delta.sum(axis=1, keepdims=True)
            if layer:
                a = activations[layer]
                delta = (self.weights[layer].T @ delta) * a * (1 - a)
        return loss, dw, db

    def train_epoch(self, x, y, learning_rate, batch_size, rng):
        x = self._inputs(x)
        if y.shape != (self.sizes[-1], x.shape[1]) or not np.all(np.isfinite(y)):
            raise ValueError('targets must match the number of classes and examples')
        if not np.isfinite(learning_rate) or learning_rate <= 0 or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('learning rate and integer batch size must be positive')
        order = rng.permutation(x.shape[1])
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            loss, dw, db = self.loss_and_gradients(x[:, indices], y[:, indices])
            for w, b, grad_w, grad_b in zip(self.weights, self.biases, dw, db):
                w -= learning_rate * grad_w
                b -= learning_rate * grad_b
            total_loss += loss * len(indices)
        return total_loss / len(order)

    def evaluate(self, x, labels, batch_size=1000):
        x = self._inputs(x)
        targets = one_hot(labels, self.sizes[-1])
        if targets.shape[1] != x.shape[1] or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('labels must match examples and batch size must be positive')
        squared_error, correct = 0.0, 0
        for start in range(0, x.shape[1], batch_size):
            end = start + batch_size
            output = self.predict(x[:, start:end])
            squared_error += np.sum((output - targets[:, start:end]) ** 2)
            correct += np.count_nonzero(output.argmax(axis=0) == labels[start:end])
        return {'mse': float(squared_error / targets.size), 'accuracy': float(correct / x.shape[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'Data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--hidden', type=int, nargs='+', default=[100, 20])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--train-limit', type=int, help='subsample training cases after reserving validation data')
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--metrics', type=Path, help='write reproducible run metadata and metrics as JSON')
    args = parser.parse_args()
    if args.epochs < 1 or args.validation_size < 1 or args.batch_size < 1 or args.learning_rate <= 0 or not np.isfinite(args.learning_rate) or any(n < 1 for n in args.hidden):
        parser.error('epochs, sizes, and finite learning rate must be positive')
    x, labels = load_mnist(args.data, 'train')
    if args.validation_size >= x.shape[1] or (args.train_limit is not None and not 1 <= args.train_limit <= x.shape[1] - args.validation_size):
        parser.error('leave training cases after validation; train-limit must fit the training partition')
    # Split only the official training set. The official test set is evaluated once, at the end.
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(x.shape[1])
    valid = order[:args.validation_size]
    train = order[args.validation_size:][:args.train_limit]
    train_x, train_labels = x[:, train], labels[train]
    valid_x, valid_labels = x[:, valid], labels[valid]
    del x
    network = Network([784, *args.hidden, 10], args.seed)
    shuffle_rng = np.random.default_rng(args.seed + 1)
    targets = one_hot(train_labels, 10)
    history = []
    for epoch in range(1, args.epochs + 1):
        training_loss = network.train_epoch(train_x, targets, args.learning_rate, args.batch_size, shuffle_rng)
        metrics = {'epoch': epoch, 'online_train_mse': training_loss, 'validation': network.evaluate(valid_x, valid_labels)}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    test_x, test_labels = load_mnist(args.data, 'test')
    result = {'seed': args.seed, 'architecture': list(network.sizes), 'learning_rate': args.learning_rate,
              'batch_size': args.batch_size, 'train_cases': len(train), 'validation_cases': len(valid),
              'test_cases': len(test_labels), 'numpy_version': np.__version__, 'history': history,
              'test': network.evaluate(test_x, test_labels)}
    print(json.dumps({'test': result['test']}))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
