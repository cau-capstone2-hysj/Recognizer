"""
handles camera object from cv2
"""
import cv2
import numpy as np


class Webcam:
    """
    an webcam object
    """

    def __init__(self) -> None:
        # from time import perf_counter

        # pt = perf_counter()
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # npt = perf_counter()
        # print(f"Webcam init took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        # npt = perf_counter()
        # print(f"Webcam set fourcc took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # npt = perf_counter()
        # print(f"Webcam set resolution took {npt - pt} seconds")
        # pt = npt
        capture.set(cv2.CAP_PROP_FPS, 60)
        # npt = perf_counter()
        # print(f"Webcam set fps took {npt - pt} seconds")

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
