"""
custum typings for the project
"""
from typing import Sequence, Union

import numpy as np

# not using numbers.Number, because Number contains complex numbers
TypeRealNumber = Union[int, float]
TypeCoord = Union[Sequence[TypeRealNumber], np.ndarray]
