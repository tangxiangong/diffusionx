from typing import Union

real = Union[float, int]


def check_transform(value: real) -> float:
    if isinstance(value, int):
        return float(value)
    elif isinstance(value, float):
        return value
    else:
        raise ValueError(f"{value} must be a number")
