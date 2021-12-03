from typing import NamedTuple

import numpy as np

from ..landmark.holisticlandmarks import HolisticLandmarks


class MediapipeResult(NamedTuple):
    image: np.ndarray
    rawImage: np.ndarray
    results: HolisticLandmarks
    rawResults: type
