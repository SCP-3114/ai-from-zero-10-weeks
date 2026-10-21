import numpy as np
from numpy.typing import NDArray


def matrix_report(
    matrix_a: NDArray[np.float64],
    matrix_b: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | float]:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"矩阵形状不一致：{matrix_a.shape} 和 {matrix_b.shape}"
        )

    return {
        "addition": matrix_a + matrix_b,
        "elementwise_multiplication": matrix_a * matrix_b,
        "matrix_multiplication": matrix_a @ matrix_b,
        "transpose": matrix_a.T,
        "determinant": float(np.linalg.det(matrix_a)),
    }


def main() -> None:
    matrix_a = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    matrix_b = np.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    results = matrix_report(matrix_a, matrix_b)

    for name, value in results.items():
        print(f"\n{name}:")
        print(value)


if __name__ == "__main__":
    main()
