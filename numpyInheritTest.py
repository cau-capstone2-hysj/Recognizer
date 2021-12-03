import numpy as np


class Vector(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def addr(self, other):
        return Vector(self + other)


def main():
    v1 = Vector([1.1, 2, 3])
    v2 = Vector([4, 5, 6])
    print(v1.addr(v2))


if __name__ == "__main__":
    main()
