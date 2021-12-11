"""
testing recongize modules by webcam
"""

import logging
from pprint import pprint

import cv2

# from components.landmarkHandler import HolisticLandmarks, Vector3d
# from components.mediapipeHandler import MP_Holistic
# from components.myArm import MyArm
from components.arm import MyArm

# mph = MP_Holistic("./sampleImg/sample1.jpg")
# mph = MP_Holistic()
# mph.showResult()
# mpr = mph.getMPResult()
# img, result = mpr.image, mpr.results
# while not (result.hand["left"] or result.hand["right"]):
#     print("retry..")
# pprint(result)

logging.getLogger().setLevel(logging.INFO)
ma = MyArm(is_rightarm=True, vis_threshold=0.70)
while 1:
    r = ma.process()
    image = cv2.putText(
        r.image,
        ", ".join(map(lambda x: str(round(x, 2)), r.theta)),
        (500, 100),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 255, 0),
        3,
    )
    cv2.imshow("MediaPipe Holistic", r.image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
