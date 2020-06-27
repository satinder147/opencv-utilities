import cv2
import sys
import numpy as np
sys.path.append(sys.path[0][:(sys.path[0].rfind("/"))])
from opencv_utilities import bounding_rect


def test_text_inside_box():
    text = 'I am Satinder Singh engineering student from India.'
    img_height = 480
    img_width = 640
    for w in range(50, img_width-20, 30):
        for h in range(50, img_height-20, 30):
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            result = bounding_rect.get_image(10, 10, w, h, img, text, 0.1, 0.1, 2, align="center")
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            if contours:
                cv2.drawContours(mask, contours, -1, 255, -1)
            mask2 = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.rectangle(mask2, (10, 10), (w+10, h+10), 255, -1)
            mask2 = cv2.bitwise_not(mask2)
            back = cv2.bitwise_and(gray, mask, mask=mask)
            back = cv2.bitwise_and(back, back, mask=mask2)
            if np.where(back > 0)[0].shape[0] > 10:
                print(w, h)
                assert False
            else:
                assert True


