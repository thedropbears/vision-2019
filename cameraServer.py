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

    

#Magic Numbers
lowerGreen = (50, 120, 130)  #Our Robot's Camera
higherGreen = (100, 220, 220)
minContourArea = 10
angleOffset = 14
rightAngleSize = -14
leftAngleSize = -75.5
screenSize = (320, 240)
distance_away = 110

#Initialisation
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


#Our code begins here
def startCamera(config):
    """Start running the camera."""
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera


#Process Functions
def getRetroPos(img, display=False, distance_away=distance_away):
    """Function for finding retro-reflective tape"""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV to make the mask easier
    mask = cv2.inRange(hsv, lowerGreen, higherGreen) #Create a mask of everything in between the greens
    mask = cv2.dilate(mask, None, iterations=1) #Expand the mask to allow for further away tape

    contours = cv2.findContours(mask, 1, 2)[1] #Find the contours
    
    if len(contours) > 1: #Get contours with area above magic number 10 and append its smallest rectangle
        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > minContourArea:
                rects.append(cv2.minAreaRect(cnt))
    
        pairs = []
        leftRect = None
        for rect in sorted(rects, key=lambda x:x[0]): #Get rectangle pairs
            if (leftAngleSize - angleOffset) < rect[2] < (leftAngleSize + angleOffset):
                leftRect = rect
            elif (rightAngleSize - angleOffset) < rect[2] < (rightAngleSize + angleOffset):
                if leftRect:
                    pairs.append((leftRect, rect))
                    leftRect = None
                    
        if len(pairs) >= 1:
            closestToMiddle = min(pairs, key = lambda x:abs((x[0][0][0]+x[1][0][0]) - screenSize[0]))
        else:
            return False, math.nan, img, mask

        boxed_points = [np.int0(cv2.boxPoints(closestToMiddle[0])), np.int0(cv2.boxPoints(closestToMiddle[1]))]
        mid_points = (max(boxed_points[0], key=lambda x:x[0]), min(boxed_points[1], key=lambda x:x[0]))
        center_point = int(np.mean([mid_points[0][0], mid_points[1][0]]))
        (x,y),radius = cv2.minEnclosingCircle(np.array(boxed_points).reshape(-1, 2))

        if display: #Create the annotated display if display is True
            img = cv2.line(img, (160, 0), (160, 240), (255, 0, 0), thickness=1)
            center = (int(x), int(y))
            for pair in pairs:
                if pair == closestToMiddle:
                    img = cv2.drawContours(img, [boxed_points[0]], 0, (0, 255, 0), thickness=2)
                    img = cv2.drawContours(img, [boxed_points[1]], 0, (0, 255, 0), thickness=2)
                    img = cv2.circle(img, center, int(radius), (0, 255, 0))
                else:
                    img = cv2.drawContours(img, [np.int0(cv2.boxPoints(pair[0]))], 0, (0, 0, 255), thickness=2)
                    img = cv2.drawContours(img, [np.int0(cv2.boxPoints(pair[1]))], 0, (0, 0, 255), thickness=2)

        return radius>distance_away, -(((center_point/screenSize[0])*2)-1), img, mask
    return False, math.nan, img, mask


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    cameraConfigs = readConfig()

    # start NetworkTables
    NetworkTables.initialize(server='10.47.74.2')

    NetworkTables.setUpdateRate(1)
    nt = NetworkTables.getTable('/vision')
    entry_tape_angle = nt.getEntry('target_tape_error')
    entry_game_piece = nt.getEntry('game_piece')
    entry_outake = nt.getEntry('within_deposit_range')

    # start cameras
    cameras = []
    for cameraConfig in cameraConfigs:
        cameras.append(startCamera(cameraConfig))

    
    cargo_sink = cameras[0][0].getVideo(camera=cameras[0][1])
    hatch_sink = cameras[1][0].getVideo(camera=cameras[1][1])

    source = cameras[0][0].putVideo('Driver_Stream', 320, 240)
    source2 = cameras[1][0].putVideo('mask', 320, 240)

    frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    game_piece = 0 #0 = hatch, 1 = cargo

    while True:
        game_piece = entry_game_piece.getBoolean(0)
        if not game_piece:
            frame_time, frame = hatch_sink.grabFrameNoTimeout(image=frame)
            if frame_time == 0:
                print(hatch_sink.getError(), file=sys.stderr)
                source.notifyError(hatch_sink.getError())
                outtake = False
                percent = math.nan
            else:
                outake, percent, image, mask = getRetroPos(frame, True, distance_away=distance_away)
        else:
            frame_time, frame = cargo_sink.grabFrameNoTimeout(image=frame)
            if frame_time == 0:
                print(cargo_sink.getError(), file=sys.stderr)
                source.notifyError(cargo_sink.getError())
                outtake = False
                percent = math.nan
            else:
                outake, percent, image, mask = getRetroPos(frame, True, distance_away=distance_away)

        source.putFrame(image)
        source2.putFrame(mask)
        entry_tape_angle.setNumber(percent)
        entry_outake.setBoolean(outake)
        NetworkTables.flush()
