from cameraServer import getRetroPos
import cv2
import numpy as np
import math
import csv
import sys


def test_images():
    acceptable_error = 0.08
    with open("samples/target.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        csvlist = list(csvreader)
        for row in csvlist:
            if row:
                frame = cv2.imread("samples/" + row[0])
                result = getRetroPos(frame, True, hsv, mask)
                data = (float(row[1]), float(row[2]))  # Distance, Offset
                if math.isnan(result[1]):
                    assert math.isnan(data[0]) and math.isnan(result[1])
                    assert math.isnan(data[1]) and math.isnan(result[2])
                    continue
                assert (
                    data[0] - acceptable_error < result[1] < data[0] + acceptable_error
                )
                assert (
                    data[1] - acceptable_error < result[2] < data[1] + acceptable_error
                )


def test_video(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() == False:
        print("Error opening video stream or file", file=sys.stderr)
        sys.exit()
    for _ in range(2350):
        _, _ = cap.read()
    while True:
        ret, frame = cap.read()
        result = getRetroPos(frame, True, hsv, mask)
        if ret:
            cv2.imshow("frame", result[0])
            print(result[1], result[2])
            cv2.waitKey(0)
        else:
            break

if __name__ == "__main__":
    screenSize = (320, 240)
    hsv = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
    mask = np.zeros(shape=(screenSize[1], screenSize[0]), dtype=np.uint8)
    test_images()

