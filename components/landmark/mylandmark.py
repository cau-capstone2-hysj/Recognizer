"""
extends Vector3d to include a visibility value
(need to be depricated)
"""
from typing import Union

from .vector3d import Vector3d


class MyLandmark(Vector3d):
    def __init__(self, *args):
        if len(args) == 4:
            __x, __y, __z, __vis = args
            super().__init__(__x, __y, __z)

        elif len(args) == 3:
            __x, __y, __z = args
            super().__init__(__x, __y, __z)
            __vis = 1.0

        elif len(args) == 2:
            __x, __y = args
            __vis = 1.0
            super().__init__(__x, __y, 0)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
        self.__vis = __vis

    @property
    def vis(self) -> Union[int, float]:
        return self.__vis

    def __str__(self) -> str:
        return f"{super().__str__()}, {self.__vis}"

    def __repr__(self) -> str:
        return self.__str__()
