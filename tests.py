from cameraServer import getRetroPos
import cv2
import numpy as np

lowerGreen = (50, 120, 130)
higherGreen = (100, 220, 220)
screenSize = (320, 240)

images = [
    ["350mm.png", (0.33, 0.37), (-0.05, 0.05)],
    ["530mm.png", (0.5, 0.56), (-0.05, 0.05)],
    ["740mm.png", (0.71, 0.77), (-0.05, 0.05)],
    ["830-300mm.png", (0.8, 0.86), (-0.3, -0.2)],
    ["860-90mm.png", (0.83, 0.89), (-0.15, -0.05)],
    ["1210mm.png", (1.16, 1.26), (-0.1, 0.1)],
]

hsv = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
mask = np.zeros(shape=(screenSize[1], screenSize[0]), dtype=np.uint8)
frame = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
for image in images:
    values = getRetroPos(cv2.imread('.\\samples\\' + image[0]), True, hsv, mask)
    print(values[1], values[2])
    assert image[1][0] < values[1] < image[1][1]
    assert image[2][0] < values[2] < image[2][1]
print("tests passed")