"""
Defines Vector3d class which inherits numpy ndarray
and extends following feature:

unitvector, anglebtw, distbtw
"""

from typing import List, Tuple, Union

import numpy as np


def check_arguments(args: tuple) -> None:
    if len(args) == 1:
        if not isinstance(args[0], (tuple, list, np.ndarray)):
            raise TypeError(f"input array must be list or ndarray, not {type(args[0])}")
        if len(args[0]) not in {2, 3}:
            raise ValueError("input array must be 3 or 2 elements")
    elif len(args) == 2 and not all(isinstance(arg, (int, float)) for arg in args):
        raise TypeError("Vector cooridnates must be int or float")
    elif len(args) == 3 and not all(isinstance(arg, (int, float)) for arg in args):
        raise TypeError("Vector cooridnates must be int or float")
    elif len(args) > 3:
        raise ValueError(f"Invalid number of arguments: {len(args)} > 3")


class Vector3d(np.ndarray):
    """
    unitvector: return unit vector of self
    anglebtw: return angle between self and other
    distbtw: return distance between self and other
    """

    def __new__(
        cls,
        *args: Union[
            Tuple[Union[int, float]], List[Union[int, float]], np.ndarray, int, float
        ],
    ):
        check_arguments(args)
        input_array = np.squeeze(np.array(args))
        if len(input_array) == 2:
            input_array = np.append(input_array, [0.0])
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(
        self,
        *args: Union[
            Tuple[Union[int, float]], List[Union[int, float]], np.ndarray, int, float
        ],
    ):
        self.__x = self[0]
        self.__y = self[1]
        self.__z = self[2]

    def unitvector(self) -> "Vector3d":
        ans = self.np / np.linalg.norm(self.np)
        return Vector3d([ans[0], ans[1], ans[2]])

    def anglebtw(self, other: "Vector3d") -> float:
        uv_me = self.unitvector()
        uv_other = other.unitvector()
        product = np.dot(uv_me.np, uv_other.np)
        product = min(max(product, -1.0), 1.0)
        angle_rad = np.arccos(product)
        return np.rad2deg(angle_rad)

    def distbtw(self, other: "Vector3d") -> float:
        return np.linalg.norm(self.np - other.np)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Vector3d):
            return np.isclose(self.np, __o.np).all()
        return False

    @property
    def x(self) -> np.float64:
        return self.__x

    @property
    def y(self) -> np.float64:
        return self.__y

    @property
    def z(self) -> np.float64:
        return self.__z

    @property
    def np(self) -> np.ndarray:
        return self
