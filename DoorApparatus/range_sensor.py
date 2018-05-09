
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
TRIG=22
ECHO=24

print("Distance measurement in prgoress")

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG,False)
print("Waiting for sensor to settle")

time.sleep(2)
GPIO.output(TRIG,True)
time.sleep(0.0001)
GPIO.output(TRIG,False)
import sys
t=time.time()

while GPIO.input(ECHO) == 0:
    pulse_start=time.time()
    if t - time.time() >10:
        print("Fail")
        sys.exit()
        break
t=time.time()		
while GPIO.input(ECHO) == 1:
    pulse_end = time.time()
    if t - time.time() >10:
        print("Fail")
        sys.exit()
        break

pulse_duration = pulse_end - pulse_start
		
distance = pulse_duration*17150
distance = round(distance,2)

print("Distance", distance, "cm")
GPIO.cleanup()