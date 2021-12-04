from typing import NamedTuple

import numpy as np

from ..landmark.holisticlandmarks import HolisticLandmarks


class MediapipeResult(NamedTuple):
    image: np.ndarray
    raw_image: np.ndarray
    results: HolisticLandmarks
    raw_results: type
