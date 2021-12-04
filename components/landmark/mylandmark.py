"""
extends Vector3d to include a visibility value
"""
from typing import Optional, Union

import numpy as np

from ..custom_typings import TypeCoord, TypeRealNumber
from .vector3d import Vector3d


class MyLandmark:
    """
    Landmark class to store the landmark coordinates and visibility
    """

    def __init__(
        self,
        coordination: Union[TypeRealNumber, TypeCoord],
        visibility: Optional[float] = None,
    ):
        self.__coord = Vector3d(coordination)
        self.__vis = visibility if visibility else 1.0

    @property
    def vis(self) -> Union[int, float]:
        return self.__vis

    @property
    def coord(self) -> Vector3d:
        return self.__coord

    @property
    def x(self) -> np.float64:
        return self.__coord.x

    @property
    def y(self) -> np.float64:
        return self.__coord.y

    @property
    def z(self) -> np.float64:
        return self.__coord.z

    def __str__(self) -> str:
        return f"{self.__coord}, {self.__vis}"

    def __repr__(self) -> str:
        return self.__str__()
