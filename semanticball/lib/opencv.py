import cv2
import numpy as np

def capture_size_info(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height

def capture_edge(output):
    lower_blue = np.array([100, 0, 0], dtype=np.uint8)
    upper_blue = np.array([255, 100, 100], dtype=np.uint8)
    mask = cv2.inRange(output, lower_blue, upper_blue)
    blue_part = cv2.bitwise_and(output, output, mask=mask)
    
    gray = cv2.cvtColor(blue_part, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
