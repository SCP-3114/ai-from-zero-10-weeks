from typing import Any


def summarize_scores(scores: list[float]) -> dict[str, float]:
    """计算一组成绩的基本统计量。"""
    if not scores:
        raise ValueError("成绩列表不能为空")

    return {
        "count": float(len(scores)),
        "sum": sum(scores),
        "mean": sum(scores) / len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
    }


def describe_student(student: tuple[str, int]) -> str:
    name, age = student
    return f"{name} is {age} years old."


def print_mapping(data: dict[str, Any]) -> None:
    for key, value in data.items():
        print(f"{key}: {value}")


def main() -> None:
    scores = [82.0, 91.5, 76.0, 88.5]
    student = ("Alice", 20)

    statistics = summarize_scores(scores)

    print(describe_student(student))
    print_mapping(statistics)


if __name__ == "__main__":
    main()
