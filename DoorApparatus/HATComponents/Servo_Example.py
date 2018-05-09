#!/usr/bin/python

from Adafruit_PWM_Servo_Driver import PWM
import time

# ===========================================================================
# Example Code
# ===========================================================================

# Initialise the PWM device using the default address
pwm = PWM(0x70)
# Note if you'd like more debug output you can instead run:
pwm = PWM(0x70, debug=True)

servoMin = 4000  # Min pulse length out of 4096
servoMax = 25  # Max pulse length out of 4096

def setServoPulse(channel, pulse):
  pulseLength = 1000000                   # 1,000,000 us per second
  pulseLength //= 60                       # 60 Hz
  print("%d us per period" % pulseLength)
  pulseLength //= 4096                     # 12 bits of resolution
  print("%d us per bit" % pulseLength)
  pulse *= 20
  pulse //= pulseLength
  pwm.setPWM(channel, 0, pulse)

pwm.setPWMFreq(60)                        # Set frequency to 60 Hz
while (True):
  # Change speed of continuous servo on channel O
  print("open door:")
  #pwm.setPWM(0, 0, 4095//2 )
  setServoPulse(0,1)
  time.sleep(3)

  print("Close door:")
  #pwm.setPWM(0,0,4095//2)
  setServoPulse(0,10)
  #print("Second pulse:",servoMax)
  #pwm.setPWM(0, servoMin, servoMax)
  time.sleep(3)

  print("Stop")
  pwm.setPWM(0,4096,0)
  break



