#!/usr/bin/env python3
import json
import sys
import numpy as np
import cv2
import math
import time
from collections import namedtuple
from cscore import CameraServer
from networktables import NetworkTables


# Magic Numbers
lowerGreen = (50, 120, 130)  # Our Robot's Camera
higherGreen = (100, 220, 220)
minContourArea = 10
angleOffset = 10
rightAngleSize = -14
leftAngleSize = -75.5
screenX = 320
screenY = 240
screenSize = (screenX, screenY)
distance_away = 110
realTapeDistance = 0.2  # metres between closest tape points
focal_length = 325

# Initialisation
configFile = "/boot/frc.json"

CameraConfig = namedtuple("CameraConfig", ["name", "path", "config"])


def readCameraConfig(config):
    """Read single camera configuration."""
    return CameraConfig(config["name"], config["path"], config)


def readConfig():
    """Read configuration file."""
    # parse file

    with open(configFile) as f:
        j = json.load(f)

    # cameras
    cameras = j["cameras"]
    cameras = [readCameraConfig(camera) for camera in cameras]

    return cameras


# Our code begins here
def startCamera(config):
    """Start running the camera."""
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera


# Process Functions
def getDistance(boxes):
    if boxes is None:
        return math.nan, math.nan
    Lpoint = max(boxes[0], key=lambda x: x[0])
    Rpoint = min(boxes[1], key=lambda x: x[0])
    width = abs(Lpoint[0] - Rpoint[0])
    mid = (Rpoint[0] + Lpoint[0]) / 2
    distance_from_center = mid - screenX / 2
    offset = getOffset(width, distance_from_center)
    if width > 0:
        dist = (realTapeDistance * focal_length) / width
        return dist, offset
    else:
        return math.nan, offset


def getOffset(width, x):
    # if width = 20cm then what is x in cm
    offset = x / (width / (realTapeDistance))
    return -offset


def createAnnotatedDisplay(
    frame: np.array, pairs: list, closestToMiddle: tuple, circle: tuple
):
    img = cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), thickness=1)
    for pair in pairs:
        if (pair[0][1][0] == closestToMiddle[0][0]).all():
            colour = (0, 255, 0)
            img = cv2.circle(
                img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), colour
            )
        else:
            colour = (0, 0, 255)
        for tape in pair:
            img = cv2.drawContours(
                img, [np.int0(tape[1])], 0, colour, thickness=2
            )
    return img


def getRetroPos(frame: np.array, annotated: bool = False):
    """Function for finding retro-reflective tape"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Convert to HSV to make the mask easier
    mask = cv2.inRange(hsv, lowerGreen, higherGreen)
    # Create a mask of everything in between the greens
    mask = cv2.dilate(mask, None, iterations=1)
    # Expand the mask to allow for further away tape

    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contours

    if len(contours) <= 1:
        # Get contours with area above magic number 10 and append its smallest rectangle
        return frame, math.nan, math.nan

    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > minContourArea:
            rects.append(cv2.minAreaRect(cnt))
    boxed_and_angles = []
    for rect in rects:
        if math.isclose(rect[2], leftAngleSize, abs_tol=angleOffset):
            boxed_and_angles.append([False, np.array(cv2.boxPoints(rect))])
        elif math.isclose(rect[2], rightAngleSize, abs_tol=angleOffset):
            boxed_and_angles.append([True, np.array(cv2.boxPoints(rect))])

    pairs = []
    leftRect = None
    for rect in sorted(
        boxed_and_angles, key=lambda x: max(x[1][:, 0]) if x[0] else min(x[1][:, 0])
    ):  # Get rectangle pairs
        if not rect[0]:
            leftRect = rect
        elif leftRect:
            pairs.append((leftRect, rect))
            leftRect = None

    if len(pairs) < 1:
        return frame, math.nan, math.nan

    closestToMiddle = list(min(
        pairs, key=lambda x: abs(np.mean([x[0][1][:,0] + x[1][1][:,0]]) - screenSize[0])
    ))
    closestToMiddle = [closestToMiddle[0][1], closestToMiddle[1][1]]

    (x, y), radius = cv2.minEnclosingCircle(np.array(closestToMiddle).reshape(-1, 2))

    if annotated:
        img = createAnnotatedDisplay(frame, pairs, closestToMiddle, ((x, y), radius))

    dist, offset = getDistance(closestToMiddle)
    return (
        img,
        dist,
        offset,
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    cameraConfigs = readConfig()

    # start NetworkTables
    NetworkTables.initialize(server="10.47.74.2")

    NetworkTables.setUpdateRate(1)
    nt = NetworkTables.getTable("/vision")
    ping = nt.getEntry("ping")
    raspi_pong = nt.getEntry("raspi_pong")
    rio_pong = nt.getEntry("rio_pong")

    entry_game_piece = nt.getEntry("game_piece")
    entry_dist = nt.getEntry("fiducial_x")
    entry_offset = nt.getEntry("fiducial_y")
    entry_fiducial_time = nt.getEntry("fiducial_time")

    # start cameras
    cameras = []
    for cameraConfig in cameraConfigs:
        cameras.append(startCamera(cameraConfig))

    cargo_sink = cameras[0][0].getVideo(camera=cameras[0][1])
    hatch_sink = cameras[1][0].getVideo(camera=cameras[1][1])

    source = cameras[0][0].putVideo("Driver_Stream", screenX, screenY)
    # source2    = cameras[1][0].putVideo('mask', 320, 240)

    frame = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
    image = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
    hsv = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
    mask = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)
    img = np.zeros(shape=(screenSize[1], screenSize[0], 3), dtype=np.uint8)

    game_piece = 0  # 0 = hatch, 1 = cargo
    old_ping_time = 0
    while True:
        ping_time = ping.getNumber(0)
        if abs(ping_time - old_ping_time) > 0.00000001:
            raspi_pong.setNumber(time.monotonic())
            rio_pong.setNumber(ping_time)
            old_ping_time = ping_time
        game_piece = entry_game_piece.getBoolean(0)
        fiducial_time = time.monotonic()
        if not game_piece:
            frame_time, frame = hatch_sink.grabFrameNoTimeout(image=frame)
            if frame_time == 0:
                print(hatch_sink.getError(), file=sys.stderr)
                source.notifyError(hatch_sink.getError())
                outtake = False
                percent = math.nan
            else:
                image, dist, offset = getRetroPos(frame, True)
        else:
            frame_time, frame = cargo_sink.grabFrameNoTimeout(image=frame)
            if frame_time == 0:
                print(cargo_sink.getError(), file=sys.stderr)
                source.notifyError(cargo_sink.getError())
                outtake = False
                percent = math.nan
            else:
                image, dist, offset = getRetroPos(frame, True)
        source.putFrame(image)
        # source2.putFrame(mask)
        if not math.isnan(dist):
            entry_dist.setNumber(dist)
            entry_offset.setNumber(offset)
            entry_fiducial_time.setNumber(fiducial_time)
        NetworkTables.flush()
