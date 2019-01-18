#!/usr/bin/env python3
import json
import time
import sys
import numpy as np
import cv2
import math

#Magic Numbers
lowerGreen = (38, 125, 100) #Our Robot's Camera
higherGreen = (52, 255, 180)
sample_lowerGreen = (30, 177, 80) #FRC sample images
sample_higherGreen = (150, 255, 255)
lowerTapeColour = (0, 0, 0) #TODO: Ground Tape filter colours
upperTapeColour = (360, 255, 255) #TODO: Ground Tape filter colours 
minContourArea = 10
angleOffset = 13
rightAngleSize = -14.5
leftAngleSize = -75.5
screenSize = (320, 240)

#Initialisation
configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

"""Read single camera configuration."""
def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

"""Report parse error."""
def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

"""Read configuration file."""
def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

"""Start running the camera."""
def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera



#Process Functions

def getRetroPos(img, display=False, sample=False):
    """Function for finding retro-reflective tape a"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV to make the mask easier
    if sample:
        mask = cv2.inRange(hsv, sample_lowerGreen, sample_higherGreen) #Create a mask of everything in between the greens
    else:
        mask = cv2.inRange(hsv, lowerGreen, higherGreen)
    mask = cv2.dilate(mask, None, iterations=2) #Expand the mask to allow for further away tape

    contours = cv2.findContours(mask, 1, 2)[-2] #Find the contours
    
    if len(contours) > 1: #Get contours with area above magic number 10 and append its smallest rectangle
        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > minContourArea:
                rects.append(cv2.minAreaRect(cnt))
        rects.sort(key=lambda x:x[2])
        
        pairs = []
        leftRect = None
        for i, rect in enumerate(sorted(rects, key=lambda x:x[0])): #Get rectangle pairs
            if rect[2] < (leftAngleSize + angleOffset) and rect[2] > (leftAngleSize - angleOffset):
                leftRect = rect
            elif rect[2] < (rightAngleSize + angleOffset) and rect[2] > (rightAngleSize - angleOffset):
                if leftRect:
                    pairs.append((leftRect, rect))
                    leftRect = None
                    
        if len(pairs) >= 1:
            closestToMiddle = min(pairs, key = lambda x:abs(x[0][0][0] - screenSize[0]/2))
        else:
            return None, None, img

        circleContours = list(np.int0(cv2.boxPoints(closestToMiddle[0])))
        circleContours.extend(list(np.int0(cv2.boxPoints(closestToMiddle[1]))))
        circleContours = np.array(circleContours)
        (x,y),radius = cv2.minEnclosingCircle(circleContours)
        radius = int(radius)
        center = (int(x), int(y))

        angle = (closestToMiddle[0][1][0] * closestToMiddle[0][1][1]) / (closestToMiddle[1][1][0] * closestToMiddle[1][1][1])
        
        if display == True: #Create the annotated display if display is True
            for pair in pairs:
                if pair == closestToMiddle:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 255, 0), thickness=2)
                    img = cv2.circle(img, center, radius, (0, 255, 0))
                else:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 0, 255))

            return angle, -(((x/screenSize[0])*2)-1), img
    return None, None, img

def getGroundPos(img, sample=False): #TODO: Ground Tape Function
    return None, None, None #Distance from x, distance from y, angle of line
    
#if False:
if __name__ == "__main__":
    from cscore import CameraServer, VideoSource
    from networktables import NetworkTables
    
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    NetworkTables.initialize(server='10.47.74.2')

    
    nt = NetworkTables.getTable('/vision')
    entry = nt.getEntry('target_tape_error')
    entry_angle = nt.getEntry('angle')

    # start cameras
    cameras = []
    for cameraConfig in cameraConfigs:
        cameras.append(startCamera(cameraConfig))

    sink = cameras[0][0].getVideo(camera=cameras[0][1])
    source = cameras[0][0].putVideo('cv', 320, 240)

    frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    while True:
        time1, frame = sink.grabFrameNoTimeout(image=frame)
        angle, percent, image = getRetroPos(frame, True)

        if not percent:
            percent = 999
        if not angle:
            angle = 999
        source.putFrame(image)
        entry.setNumber(percent)
        entry_angle.setNumber(angle)
        NetworkTables.flush()
