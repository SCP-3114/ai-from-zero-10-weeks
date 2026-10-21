import numpy as np

from exercises.matrix_operations import matrix_report


def test_matrix_addition() -> None:
    matrix_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix_b = np.array([[5.0, 6.0], [7.0, 8.0]])

    results = matrix_report(matrix_a, matrix_b)

    expected = np.array([[6.0, 8.0], [10.0, 12.0]])
    np.testing.assert_array_equal(results["addition"], expected)


def test_matrix_multiplication() -> None:
    matrix_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix_b = np.array([[5.0, 6.0], [7.0, 8.0]])

    results = matrix_report(matrix_a, matrix_b)

    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_array_equal(
        results["matrix_multiplication"],
        expected,
    )
