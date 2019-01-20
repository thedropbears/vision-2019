#!/usr/bin/env python3
import json
import time
import sys
import numpy as np
import cv2
import math

from cscore import CameraServer, VideoSource
from networktables import NetworkTables
    

#Magic Numbers
lowerGreen = (38, 125, 100) #Our Robot's Camera
higherGreen = (52, 255, 180)
sampleLowerGreen = (30, 177, 80) #FRC sample images
sampleHigherGreen = (150, 255, 255)
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

def readCameraConfig(config):
    """Read single camera configuration."""
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
        parseError("camera '" + str(cam.name) + "': could not read path")
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

def parseError(error_str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + error_str, file=sys.stderr)

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '" + str(configFile) + "': " + str(err), file=sys.stderr)
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
        nt_str = j["ntmode"]
        if nt_str.lower() == "client":
            server = False
        elif nt_str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '" + str(nt_str))

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



#Our code begins here
def startCamera(config):
    """Start running the camera."""
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)
    camera.setConfigJson(json.dumps(config.config))
    return cs, camera



#Process Functions
def getRetroPos(img, display=False, sample=False):
    """Function for finding retro-reflective tape a"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV to make the mask easier
    if sample:
        mask = cv2.inRange(hsv, sampleLowerGreen, sampleHigherGreen) #Create a mask of everything in between the greens
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
            return float("inf"), float("inf"), img

        circleContours = list(np.int0(cv2.boxPoints(closestToMiddle[0])))
        circleContours.extend(list(np.int0(cv2.boxPoints(closestToMiddle[1]))))
        circleContours = np.array(circleContours)
        (x,y),radius = cv2.minEnclosingCircle(circleContours)
        radius = int(radius)
        center = (int(x), int(y))

        angle = (closestToMiddle[0][1][0] * closestToMiddle[0][1][1]) / (closestToMiddle[1][1][0] * closestToMiddle[1][1][1])
        
        if display: #Create the annotated display if display is True
            for pair in pairs:
                if pair == closestToMiddle:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 255, 0), thickness=2)
                    img = cv2.circle(img, center, radius, (0, 255, 0))
                else:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 0, 255))

            return angle, -(((x/screenSize[0])*2)-1), img
    return float("inf"), float("inf"), img

def getGroundPos(img, sample=False): #TODO: Ground Tape Function
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, (50), (255))

    contours = cv2.findContours(mask, 1, 2)[-2]

    if len(contours) >= 1:
        contours.sort(key=cv2.contourArea)
        cnt = contours[-1]
        if cv2.contourArea(cnt) > 100:
            rect = cv2.minAreaRect(cnt)
            boxpoints = np.int0(cv2.boxPoints(rect))
            img = cv2.drawContours(img, [boxpoints], 0, (255))

            inner_points = []
            for point in boxpoints:
                outside = False
                for i in range(2):
                    if not 5 < point[i] < gray.shape[not i] - 5:
                        outside = True
                if not outside:
                    inner_points.append(point)
                
            
            #Cases:
            if len(inner_points) == 2: #If 2 points are on the screen, give their average
                avrPoints = np.average(inner_points, axis=0)
                x_float = -((avrPoints[0] / (gray.shape[1]/2))-1)
                y_float = -((avrPoints[1] / (gray.shape[0]/2))-1)
                orderedPoints = sorted(inner_points, key=lambda x:x[0])
                distance_x = orderedPoints[1][0] - orderedPoints[0][0]
                distance_y = orderedPoints[1][1] - orderedPoints[0][1]
                angle = -math.atan2(distance_y, distance_x)
                return angle, x_float, y_float
            elif len(inner_points) == 1: #If one point is on the screen, return its location
                x_float = -((inner_points[0][0] / (gray.shape[1]/2))-1)
                y_float = -((inner_points[0][1] / (gray.shape[0]/2))-1)
                return 0, x_float, y_float
            elif len(inner_points) == 0:
                avrPoints = np.average(boxpoints, axis=0)
                x_float = -((avrPoints[0] / (gray.shape[1]/2))-1)
                y_float = -((avrPoints[1] / (gray.shape[0]/2))-1)
                distance_1 = math.sqrt((boxpoints[0][0] - boxpoints[1][0])**2 + (boxpoints[0][1] - boxpoints[1][1]))
                distance_2 = math.sqrt((boxpoints[0][0] - boxpoints[2][0])**2 + (boxpoints[0][1] - boxpoints[2][1]))
                if distance_1 < distance_2:
                    distance_x = boxpoints[1][0] - boxpoints[0][0]
                    distance_y = boxpoints[1][1] - boxpoints[0][1]
                else:
                    distance_x = boxpoints[2][0] - boxpoints[0][0]
                    distance_y = boxpoints[2][1] - boxpoints[0][1]
                angle = -math.atan2(distance_y, distance_x)
                return angle, x_float, y_float
            else:
                avrPoints = np.average(inner_points, axis=0)
                x_float = -((avrPoints[0] / (gray.shape[1]/2))-1)
                y_float = -((avrPoints[1] / (gray.shape[0]/2))-1)
                return len(inner_points), x_float, y_float
            
            
            #Distance from x, distance from y, angle of line
    return float("inf"), float("inf"), float("inf")
 


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    NetworkTables.initialize(server='10.47.74.2')

    
    nt = NetworkTables.getTable('/vision')
    entry_x = nt.getEntry('ground_tape_x')
    entry_angle = nt.getEntry('ground_tape_angle')
    entry_y = nt.getEntry('ground_tape_y')
    entry_tape_angle = nt.getEntry('target_tape_error')
    entry_wall_ratio = nt.getEntry('wall_angle')

    # start cameras
    cameras = []
    for cameraConfig in cameraConfigs:
        cameras.append(startCamera(cameraConfig))

    ground_sink = cameras[0][0].getVideo(camera=cameras[0][1])
    retro_sink = cameras[1][0].getVideo(camera=cameras[1][1])
    source = cameras[0][0].putVideo('cv', 320, 240)

    ground_frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    retro_frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    while True:
        time1, ground_frame = ground_sink.grabFrameNoTimeout(image=ground_frame)
        time2, retro_frame = retro_sink.grabFrameNoTimeout(image=retro_frame)
        retro_angle, percent, image = getRetroPos(retro_frame, True)
        ground_angle, x, y = getGroundPos(ground_frame)

        source.putFrame(image)
        entry_x.setNumber(x)
        entry_y.setNumber(y)
        entry_angle.setNumber(ground_angle)
        entry_tape_angle.setNumber(percent)
        entry_wall_ratio.setNumber(retro_angle)
        NetworkTables.flush()
