
import time

import opencv
import boto3


dispenseFood_bool = False

def TurnOnMotionDetector():
    pass

def takeFrontPicture(filepath = './'):
    pass

def takeIndoorPicture():
    pass


def checkDistanceSensor():
    pass


def dispenseFood():
    if checkDistanceSensor:
        takeIndoorPicture('./backcamera') #take a few

def openFrontDoor():
    movementDetected = False
    running_motion_detector = TurnOnMotionDetector()

    time=time.time()
    while not movementDetected or time.time() - time < 30:
        pass

    if movementDetected:
        takeFrontPicture('./frontCamera/')  #take a few

        dispenseFood()
        

    



    
    


def closeFrontDoor():
    pass


def checkWeight():
    pass

