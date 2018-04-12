from Adafruit_MotorHAT import Adafruit_MotorHAT

import time
import atexit

mh = Adafruit_MotorHAT(0x70)
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)



class DoorMotor():
    def __init__(self, motorNumber=0):
        self.motorNumber = motorNumber
        self.mh = Adafruit_MotorHAT(0x70)
        self.m = self.mh.getMotor(1)
    def openMotorDoor(self):
        self.m.run(Adafruit_MotorHAT.BACKWARD)
        for i in range(100):
            self.m.setSpeed(i)
            time.sleep(.1)
        time.sleep(5)
        self.m.run(Adafruit_MotorHAT.RELEASE)
    def closeMotorDoor(self):
        self.m.run(Adafruit_MotorHAT.FORWARD)
        for i in range(100):
            self.m.setSpeed(i)
            time.sleep(.1)
        time.sleep(5)
        self.m.run(Adafruit_MotorHAT.RELEASE)



from Adafruit_PWM_Servo_Driver import PWM

def setServoPulse(channel, pulse):
    pulseLength = 1000000                   # 1,000,000 us per second
    pulseLength //= 60                       # 60 Hz
    print("%d us per period" % pulseLength)
    pulseLength //= 4096                     # 12 bits of resolution
    print("%d us per bit" % pulseLength)
    pulse *= 20
    pulse //= pulseLength
    pwm.setPWM(channel, 0, int(pulse))


class DoorServo():
    def __init__(self, servoAddress = 0x70, debug=True, frequency = 60):
        
        self.pwm = PWM(servoAddress, debug=debug)
        self.frequency = frequency



from Adafruit_PWM_Servo_Driver import PWM


pwm = PWM(0x70, debug=True)
frequency = 60

pwm.setPWMFreq(60)
print("Open Door:")
def openServoDoorandUnlock():
    pwm.setPWM(1,0,500);time.sleep(2) # Unlock
    pwm.setPWM(0,0,200) # Open Door
    time.sleep(1)
    pwm.setPWM(0,4096,0);pwm.setPWM(1,4096,0) #Shut off servos

def closeServoDoorandLock():
    pwm.setPWM(0,0,600) ;time.sleep(2) # Close Door
    pwm.setPWM(1,0,300) # Lock door
    time.sleep(1)
    pwm.setPWM(0,4096,0);pwm.setPWM(1,4096,0)# shut off servos
"""

## Set the door lock

def lockDoor():
    pwm.setPWM(15,1,4095//3) # Lock position
    time.sleep(3)  # Allow a second for it to lock
    pwm.setPWM(15, 4096, 0)

def unlockDoor():
    setServoPulse(15, 2)
    time.sleep(3)
    pwm.setPWM(15, 4096, 0)
"""