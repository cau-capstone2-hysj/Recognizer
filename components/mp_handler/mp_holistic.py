"""

"""
from typing import Iterator, Optional

import cv2
import numpy as np

import mediapipe as mp

from ..landmark.holisticlandmarks import HolisticLandmarks
from .mediapipe_result import MediapipeResult
from .webcam import Webcam


class MP_Holistic:
    MEDIAPIPE_HOLISTIC = mp.solutions.holistic
    MP_DRAWING = mp.solutions.drawing_utils
    MP_DRAWING_STYLES = mp.solutions.drawing_styles

    def __init__(
        self,
        imageDirectory=None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # self.isImage, self.__camera, self.__inputImg = False, None, None
        # if not imageDirectory:
        #     self.__camera = Webcam()
        # else:
        #     self.__inputImg = cv2.imread(imageDirectory, cv2.IMREAD_COLOR)
        #     self.isImage = True

        self.__nextFrame = self.__getFrameSource(imageDirectory)
        self.__holistic = MP_Holistic.MEDIAPIPE_HOLISTIC.Holistic(
            # static_image_mode=self.isImage,
            static_image_mode=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __getFrameSource(self, imageDirectory: Optional[str]) -> Iterator[np.ndarray]:
        __camera = None
        __img = None
        while 1:
            if imageDirectory:
                if __img is None:
                    __img = cv2.imread(imageDirectory, cv2.IMREAD_COLOR)
                yield __img
            else:
                if __camera is None:
                    __camera = Webcam()
                yield __camera.frame

    def getMPResult(self) -> MediapipeResult:
        bgrImage = next(self.__nextFrame)

        rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
        mpResults = self.__holistic.process(rgbImage)
        hlms = HolisticLandmarks(mpResults)
        processedBgrImage = self.__showResultsOnFrame(bgrImage.copy(), mpResults)
        return MediapipeResult(
            image=processedBgrImage,
            rawImage=bgrImage,
            results=hlms,
            rawResults=mpResults,
        )

    def __showResultsOnFrame(self, bgrImage: np.ndarray, results) -> np.ndarray:
        """
        Draws the landmarks on the frame.
        """

        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.pose_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )
        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.left_hand_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        MP_Holistic.MP_DRAWING.draw_landmarks(
            bgrImage,
            results.right_hand_landmarks,
            MP_Holistic.MEDIAPIPE_HOLISTIC.HAND_CONNECTIONS,
            landmark_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
            connection_drawing_spec=MP_Holistic.MP_DRAWING_STYLES.get_default_hand_connections_style(),
        )
        return bgrImage

    def showResult(self) -> None:
        while 1:
            r = self.getMPResult()
            cv2.imshow("MediaPipe Holistic", r.image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
