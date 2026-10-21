import csv
from pathlib import Path

import numpy as np


def read_column(file_path: Path, column: str) -> list[float]:
    values: list[float] = []

    try:
        with file_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(f"CSV中不存在列：{column}")

            for row_number, row in enumerate(reader, start=2):
                try:
                    values.append(float(row[column]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"第 {row_number} 行的 {column} 不是有效数字"
                    ) from error

    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到CSV文件：{file_path}") from error

    if not values:
        raise ValueError("CSV中没有可计算的数据")

    return values


def calculate_statistics(values: list[float]) -> dict[str, float]:
    array = np.array(values, dtype=np.float64)

    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array)),
    }


def main() -> None:
    values = read_column(Path("sample_data/scores.csv"), "python")
    statistics = calculate_statistics(values)

    for name, value in statistics.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
