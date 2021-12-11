"""
handles camera object from cv2
"""
from platform import system
from time import perf_counter

import cv2
import numpy as np


class Webcam:
    """
    an webcam object
    """

    def __init__(self) -> None:

        perf_count = perf_counter()

        capture = (
            cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if system() == "Windows"
            else cv2.VideoCapture(0)
        )
        # capture = cv2.VideoCapture(1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 60)

        print(f"Webcam launced in: {perf_counter() - perf_count}")
        self.__cap = capture

    @property
    def frame(self) -> np.ndarray:
        while 1:
            ret, frame = self.__cap.read()
            if ret:
                break
        frame = cv2.flip(frame, 1)

        return frame

    def __del__(self) -> None:
        self.__cap.release()
