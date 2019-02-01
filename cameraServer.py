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
lowerGreen = (38, 100, 80)  #Our Robot's Camera
higherGreen = (160, 255, 230)
sampleLowerGreen = (30, 177, 80)  #FRC sample images
sampleHigherGreen = (150, 255, 255)
minContourArea = 10
angleOffset = 14
rightAngleSize = -14.5
leftAngleSize = -75.5
screenSize = (320, 240)
screenWidth = 320
screenHeight = 240
distance_away = 500

halfScreenWidth = screenWidth / 2
halfScreenHeight = screenHeight / 2

#Initialisation
configFile = "/boot/frc.json"

CameraConfig = namedtuple("CameraConfig", ["name", "path", "config"])


def readCameraConfig(config):
    """Read single camera configuration."""
    return CameraConfig(config["name"], config["path"], config)


def readConfig():
    """Read configuration file."""
    # parse file

    try:
        with open(configFile) as f:
            j = json.load(f)
    except:
        exit()

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
def getRetroPos(img, display=False, distance_away=110):
    """Function for finding retro-reflective tape"""
    '''
    newimg = img[:,:,1].astype(np.int32) - img[:,:,2] - img[:,:,0]
    mask = newimg > 0
    mask = mask.astype(np.uint8)
    '''

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV to make the mask easier
    mask = cv2.inRange(hsv, lowerGreen, higherGreen) #Create a mask of everything in between the greens
    mask = cv2.dilate(mask, None, iterations=1) #Expand the mask to allow for further away tape

    contours = cv2.findContours(mask, 1, 2)[-2] #Find the contours
    
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
            closestToMiddle = min(pairs, key = lambda x:abs(x[0][0][0] - screenSize[0]/2))
        else:
            return False, math.nan, img

        circleContours = list(np.int0(cv2.boxPoints(closestToMiddle[0])))
        circleContours.extend(list(np.int0(cv2.boxPoints(closestToMiddle[1]))))
        circleContours = np.array(circleContours)
        (x,y),radius = cv2.minEnclosingCircle(circleContours)
        
        if display: #Create the annotated display if display is True
            radius = int(radius)
            center = (int(x), int(y))
            for pair in pairs:
                if pair == closestToMiddle:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 255, 0), thickness=2)
                    img = cv2.circle(img, center, radius, (0, 255, 0))
                else:
                    for tape in pair:
                        img = cv2.drawContours(img, [np.int0(cv2.boxPoints(tape))], 0, (0, 0, 255), thickness=2)

            return radius>distance_away, -(((x/screenSize[0])*2)-1), img
    return False, math.nan, img


def getGroundPos(img, sample=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, (50), (255))
    contours = cv2.findContours(mask, 1, 2)[-2]


    if len(contours) >= 1:
        cnt = max(contours, key=cv2.contourArea)
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
                distance_1 = (boxpoints[0][0] - boxpoints[1][0])**2 + (boxpoints[0][1] - boxpoints[1][1])
                distance_2 = (boxpoints[0][0] - boxpoints[2][0])**2 + (boxpoints[0][1] - boxpoints[2][1])
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
    return math.nan, math.nan, math.nan
 
 
def trackGroundTape(img):
    boxPointsOnScreen = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 100, 255)

    contours = cv2.findContours(mask, 1, 2)[-2]
    
    if len(contours) > 0:
        largestContour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largestContour) > 10:
            rect = cv2.minAreaRect(largestContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            boxPointsOnScreen = np.array(filterPoints(box, screenWidth, screenHeight))  # Get all the boxpoints on the screen.

            # Cases:
            if len(boxPointsOnScreen) == 2:  # If two points are on the screen, return the point between them.
                averages = np.mean(boxPointsOnScreen, axis=0)

                x_float = -((averages[0] / halfScreenWidth) - 1)
                y_float = -((averages[1] / halfScreenHeight) - 1)

                return x_float, y_float, findAngle((boxPointsOnScreen[0], boxPointsOnScreen[1]), oneSide=True)
            elif len(boxPointsOnScreen) == 1:  # If one point is ont the screen, return its coordinates.
                x_float = -((boxPointsOnScreen[0][0] / halfScreenWidth) - 1)
                y_float = -((boxPointsOnScreen[0][1] / halfScreenHeight) - 1)

                return 0, x_float, y_float
            else:  # For any other case, return the average point of the contour.
                averages = np.mean(box, axis=0) 

                x_float = -((averages[0] / halfScreenWidth) - 1)
                y_float = -((averages[1] / halfScreenHeight) - 1)

                return x_float, y_float, findAngle(box)

    # Return 0 if no contours found.
    return float("NaN"), float("NaN"), float("NaN")


def findAngle(points, oneSide=False):
    if oneSide:
        side = (points[0], points[1])
        opposite = side[0][0] - side[1][0]
        adjacent = side[0][1] - side[1][1]
        angle = math.atan2(opposite, adjacent)
        angle -= 1.5708  # Subtract 90 degrees in radians.
    else:
        side = findLongestSide((points[0], points[1]), (points[1], points[2]))
        opposite = side[0][0] - side[1][0]
        adjacent = side[0][1] - side[1][1]
        angle = math.atan2(opposite, adjacent)

    return math.degrees(angle)


def filterPoints(box, screenWidth, screenHeight):
    boxPointsOnScreen = []
    for point in box:
        if screenWidth - 5 > point[0] > 5 and screenHeight - 5 > point[1] > 5: #Less trailing ands, instead use multiple comparisons
            boxPointsOnScreen.append(point)
    return boxPointsOnScreen


def findLongestSide(side1, side2):
    longest_side = max((side1, side2), key=lambda a: abs(a[0][0] - a[1][0]) + abs(a[0][1] - a[1][1]))
    return longest_side
 

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    cameraConfigs = readConfig()

    # start NetworkTables
    NetworkTables.initialize(server='10.47.74.2')

    
    nt = NetworkTables.getTable('/vision')
    entry_x = nt.getEntry('ground_tape_x')
    entry_angle = nt.getEntry('ground_tape_angle')
    entry_y = nt.getEntry('ground_tape_y')
    entry_tape_angle = nt.getEntry('target_tape_error')
    entry_game_piece = nt.getEntry('game_piece')
    entry_outake = nt.getEntry('within_deposit_range')
    entry_distance = nt.getEntry('deploy_distance')

    # start cameras
    cameras = []
    for cameraConfig in cameraConfigs:
        cameras.append(startCamera(cameraConfig))

    
    cargo_retro_sink = cameras[0][0].getVideo(camera=cameras[0][1])
    cargo_ground_sink = cameras[1][0].getVideo(camera=cameras[1][1])
    hatch_retro_sink = cameras[2][0].getVideo(camera=cameras[2][1])
    hatch_ground_sink = cameras[3][0].getVideo(camera=cameras[3][1])

    source = cameras[0][0].putVideo('cv', 320, 240)

    ground_frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    retro_frame = np.zeros(shape=(screenSize[1], screenSize[0], 3))
    game_piece = 0 #0 = hatch, 1 = cargo

    while True:
        distance_away_nt = entry_distance.getNumber(110)
        game_piece = entry_game_piece.getBoolean(0)
        if not game_piece:
            _, ground_frame = hatch_ground_sink.grabFrameNoTimeout(image=ground_frame)
            _, retro_frame = hatch_retro_sink.grabFrameNoTimeout(image=retro_frame)
        else:
            _, ground_frame = cargo_ground_sink.grabFrameNoTimeout(image=ground_frame)
            _, retro_frame = cargo_retro_sink.grabFrameNoTimeout(image=retro_frame)
        outake, percent, image = getRetroPos(retro_frame, True, distance_away=distance_away_nt)
        ground_angle, x, y = getGroundPos(ground_frame)

        source.putFrame(image)
        entry_x.setNumber(x)
        entry_y.setNumber(y)
        entry_angle.setNumber(ground_angle)
        entry_tape_angle.setNumber(percent)
        entry_outake.setBoolean(outake)
        NetworkTables.flush()
