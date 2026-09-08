import struct
import numpy as np
import pytest
from main import Network, one_hot, read_idx, sigmoid


def test_backprop_matches_every_finite_difference():
    net = Network([2, 3, 2], seed=2)
    x = np.array([[0.1, -0.4, 0.7], [0.2, 0.6, -0.8]])
    y = one_hot(np.array([0, 1, 0]), 2)
    _, dw, db = net.loss_and_gradients(x, y)
    for parameter, gradient in zip(net.weights + net.biases, dw + db):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-5
            plus = net.loss_and_gradients(x, y)[0]
            parameter[index] = original - 1e-5
            minus = net.loss_and_gradients(x, y)[0]
            parameter[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / 2e-5, rel=1e-5, abs=1e-9)


def test_learning_small_problem_and_partial_batches():
    x = np.array([[0., 0., 1., 1., 0.], [0., 1., 0., 1., 0.]])
    labels = np.array([0, 1, 1, 1, 0])
    net = Network([2, 3, 2], seed=1)
    rng = np.random.default_rng(4)
    for _ in range(300):
        net.train_epoch(x, one_hot(labels, 2), 2., 3, rng)
    assert net.evaluate(x, labels, batch_size=2)['accuracy'] == 1
    assert net.evaluate(x, labels) == pytest.approx(net.evaluate(x, labels, batch_size=2))
    net.train_epoch(x, one_hot(labels, 2), 1., 100, rng)


def test_stable_extreme_sigmoid():
    with np.errstate(over='raise', invalid='raise'):
        assert np.array_equal(sigmoid(np.array([-10000., 0., 10000.])), [0., .5, 1.])


def test_idx_validates_header_and_exact_payload(tmp_path):
    path = tmp_path / 'tensor'
    path.write_bytes(b'\x00\x00\x08\x02' + struct.pack('>II', 2, 2) + bytes([1, 2, 3, 4]))
    assert read_idx(path).tolist() == [[1, 2], [3, 4]]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload'):
        read_idx(path)


def test_single_case_and_invalid_shapes():
    net = Network([2, 2])
    assert net.predict([1, 2]).shape == (2, 1)
    with pytest.raises(ValueError):
        net.predict(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        one_hot(np.array([-1]), 2)
    with pytest.raises(ValueError):
        net.train_epoch(np.ones((2, 1)), np.ones((2, 1)), 1, 0, np.random.default_rng())


def test_seed_reproduces_initialization_and_training():
    a, b = Network([2, 3, 2], 8), Network([2, 3, 2], 8)
    x, y = np.eye(2), np.eye(2)
    for network in (a, b):
        network.train_epoch(x, y, 1, 1, np.random.default_rng(9))
    for wa, wb in zip(a.weights, b.weights):
        np.testing.assert_array_equal(wa, wb)
