"""
Defines Vector3d class which inherits numpy ndarray
and extends following feature:

unitvector, anglebtw, distbtw
"""

from typing import Union

import numpy as np


class Vector3d:
    """
    unitvector: return unit vector of self
    anglebtw: return angle between self and other
    distbtw: return distance between self and other
    """

    def __init__(self, *args: Union[int, float, np.ndarray]):
        if len(args) == 1:
            self.__np = np.array(args[0], dtype=np.float64)
        elif len(args) == 2:
            self.__np = np.array([args[0], args[1], 0], dtype=np.float64)
        elif len(args) == 3:
            self.__np = np.array(args, dtype=np.float64)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        self.__x = self.__np[0]
        self.__y = self.__np[1]
        self.__z = self.__np[2]

    def unitvector(self) -> "Vector3d":
        ans = self.np / np.linalg.norm(self.np)
        return Vector3d(ans[0], ans[1], ans[2])

    def anglebtw(self, other: "Vector3d") -> float:
        uv_me = self.unitvector()
        uv_other = other.unitvector()
        product = np.dot(uv_me.np, uv_other.np)
        if product >= 1:
            return 0.0
        if product <= -1:
            return 180.0
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
        return self.__np

    def __str__(self) -> str:
        return f"({self.__x}, {self.__y}, {self.__z})"

    def __repr__(self) -> str:
        return self.__str__()
