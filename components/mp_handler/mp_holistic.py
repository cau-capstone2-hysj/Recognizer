"""

"""
from typing import Iterator, Optional

import cv2
import mediapipe as mp
import numpy as np

from ..landmark.holisticlandmarks import HolisticLandmarks
from .mediapipe_result import MediapipeResult
from .webcam import Webcam


class MP_holistic:
    """
    make infer of mediapipe model, from image or webcam
    """

    MEDIAPIPE_HOLISTIC = mp.solutions.holistic
    MP_DRAWING = mp.solutions.drawing_utils
    MP_DRAWING_STYLES = mp.solutions.drawing_styles

    def __init__(
        self,
        image_directory=None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:

        self.__nextframe = self.__get_framesource(image_directory)
        self.__holistic = MP_holistic.MEDIAPIPE_HOLISTIC.Holistic(
            # static_image_mode=self.isImage,
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __get_framesource(self, image_dir: Optional[str]) -> Iterator[np.ndarray]:
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

    def process(self) -> MediapipeResult:
        image_bgr = next(self.__nextframe)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_results = self.__holistic.process(image_rgb)
        hlms = HolisticLandmarks(raw_results)
        image_bgr_marked = self.__mark_results_on_frame(image_bgr.copy(), raw_results)
        return MediapipeResult(
            image=image_bgr_marked,
            raw_image=image_bgr,
            results=hlms,
            raw_results=raw_results,
        )

    def __mark_results_on_frame(self, image_bgr: np.ndarray, results) -> np.ndarray:
        """
        Draw landmarks on the frame.
        """

        MP_holistic.MP_DRAWING.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            MP_holistic.MEDIAPIPE_HOLISTIC.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_holistic.MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )
        MP_holistic.MP_DRAWING.draw_landmarks(
            image_bgr,
            results.left_hand_landmarks,
            MP_holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        MP_holistic.MP_DRAWING.draw_landmarks(
            image_bgr,
            results.right_hand_landmarks,
            MP_holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        return image_bgr

    def show_result(self) -> None:
        while 1:
            r = self.process()
            cv2.imshow("MediaPipe Holistic", r.image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
