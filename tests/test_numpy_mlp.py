import numpy as np

from week02_numpy_nn.mlp_classifier import MLP, cross_entropy_loss, softmax


def test_softmax_rows_sum_to_one() -> None:
    probabilities = softmax(np.array([[1000.0, 1001.0], [-1000.0, -999.0]]))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    assert np.isfinite(probabilities).all()


def test_manual_gradient_matches_numerical_gradient() -> None:
    inputs = np.array([[0.2, -0.4], [0.7, 0.1], [-0.5, 0.3]])
    labels = np.array([0, 1, 0], dtype=np.int64)
    model = MLP(input_size=2, hidden_size=3, output_size=2, seed=7)
    probabilities, cache = model.forward(inputs)
    analytical = model.backward(labels, cache)["weight2"][0, 0]

    epsilon = 1e-6
    original = model.weight2[0, 0]
    model.weight2[0, 0] = original + epsilon
    plus_loss = cross_entropy_loss(model.forward(inputs)[0], labels)
    model.weight2[0, 0] = original - epsilon
    minus_loss = cross_entropy_loss(model.forward(inputs)[0], labels)
    model.weight2[0, 0] = original
    numerical = (plus_loss - minus_loss) / (2 * epsilon)

    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)
