"""

"""
from typing import Iterator, NamedTuple, Optional

import cv2
import mediapipe as mp
import numpy as np

from ..landmark.holisticlandmarks import HolisticLandmarks
from .webcam import Webcam

MEDIAPIPE_HOLISTIC = mp.solutions.holistic
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles


def __mark_landmarks_on_image(image_bgr: np.ndarray, mp_results_raw) -> np.ndarray:
    """
    Draw landmarks on the image.
    """

    pose_connections = MEDIAPIPE_HOLISTIC.POSE_CONNECTIONS
    pose_landmarks_style = MP_DRAWING_STYLES.get_default_pose_landmarks_style()

    hand_connections = MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS
    hand_landmarks_style = MP_DRAWING_STYLES.get_default_hand_landmarks_style()
    hand_connections_style = MP_DRAWING_STYLES.get_default_hand_connections_style()

    MP_DRAWING.draw_landmarks(
        image_bgr,
        mp_results_raw.pose_landmarks,
        pose_connections,
        landmark_drawing_spec=pose_landmarks_style,
    )
    MP_DRAWING.draw_landmarks(
        image_bgr,
        mp_results_raw.left_hand_landmarks,
        hand_connections,
        landmark_drawing_spec=hand_landmarks_style,
        connection_drawing_spec=hand_connections_style,
    )
    MP_DRAWING.draw_landmarks(
        image_bgr,
        mp_results_raw.right_hand_landmarks,
        hand_connections,
        landmark_drawing_spec=hand_landmarks_style,
        connection_drawing_spec=hand_connections_style,
    )
    return image_bgr


def __get_framesource(image_dir: Optional[str]) -> Iterator[np.ndarray]:
    """
    yields frames from webcam or image directory
    if imagae_dir is None, webcam is used
    else, image_dir is used repeatedly
    """
    __camera = None
    __img = None
    while 1:
        if image_dir:
            if __img is None:
                __img = cv2.imread(image_dir, cv2.IMREAD_COLOR)
            yield __img
        else:
            if __camera is None:
                __camera = Webcam()
            yield __camera.frame


class MpResult(NamedTuple):
    """
    NamedTuple for storing the results of the MediaPipe pipeline.
    """

    marked_image: np.ndarray
    raw_image: np.ndarray
    holistic_landmarks: HolisticLandmarks
    raw_results: type


class MpHolistic:
    """
    make infer of mediapipe model, from image or webcam
    """

    def __init__(
        self,
        image_directory=None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self.__nextframe = __get_framesource(image_directory)
        self.__holistic = MEDIAPIPE_HOLISTIC.Holistic(
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self) -> MpResult:
        image_bgr = next(self.__nextframe)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        raw_results = self.__holistic.process(image_rgb)
        hlms = HolisticLandmarks(raw_results)
        image_bgr_marked = __mark_landmarks_on_image(image_bgr.copy(), raw_results)

        return MpResult(
            marked_image=image_bgr_marked,
            raw_image=image_bgr,
            holistic_landmarks=hlms,
            raw_results=raw_results,
        )

    def imshow_result(self) -> None:
        """
        show the result of the pipeline
        (for testing purposes)
        """
        while 1:
            mpr = self.process()
            cv2.imshow("MediaPipe Holistic", mpr.marked_image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
