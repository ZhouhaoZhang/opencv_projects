"""
用数字代表爪子位置
-1：从地上夹块的位置，保证爪子中点离地2.5厘米
0：运载块时的爪子位置，比-1稍高即可
1：释放1块所处高度
2：
3：
。。。。。。
6：
"""
import RPi.GPIO as GPIO
import time
from gpiozero import Servo
from time import sleep

myGPIO=17

myCorrection=0.35

maxPW=(2.0+myCorrection)/1000

minPW=(1.0)/1000

servo = Servo(myGPIO,min_pulse_width=minPW,max_pulse_width=maxPW)

 # 规定GPIO引脚
IN1 = 18      # 接PUL-
IN2 = 16      # 接PUL+
IN3 = 15      # 接DIR-
IN4 = 13      # 接DIR+


def hopen():
    servo.min()  
    print("mid")
    """
    sleep(0.5)
    
    servo.min()
    print("min")
"""



def hclose():
    """
    servo.mid()
    print("mid")
    sleep(0.5)"""
    servo.mid()
    print("max")




 
def setStep(w1, w2, w3, w4):
    GPIO.output(IN1, w1)
    GPIO.output(IN2, w2)
    GPIO.output(IN3, w3)
    GPIO.output(IN4, w4)
 
def stop():
    setStep(0, 0, 0, 0)
 
def downward(delay, steps):  
    for i in range(0, steps):
        setStep(1, 0, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(1, 0, 0, 1)
        time.sleep(delay)
 
def upward(delay, steps):  
    for i in range(0, steps):
        setStep(1, 0, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 0, 1)
        time.sleep(delay)
        setStep(0, 1, 1, 0)
        time.sleep(delay)
        setStep(1, 0, 1, 0)
        time.sleep(delay)
def lift(p1, p2):
    if p1<p2:
        if p1==-1:
            if p2 >= 1:
                p2 = p2 - 1
            upward(0.000001,(p2-p1)*10000-8000)
        else:
            if p1 == 0:
                p2 = p2 - 1
            upward(0.000001,(p2-p1)*10000)
    else:
        if p2==-1:
            if p1 >= 1:
                p1 = p1 - 1
            downward(0.000001,(p1-p2)*10000-8000)
        else:
            if p2 == 0:
                p1 = p1 - 1
            downward(0.000001,(p1-p2)*10000)
        
  
 # 爪子从p1位置移动到p2位置
   # :param p1:
    #:param p2:
   # :return:
 
def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
    GPIO.setup(IN1, GPIO.OUT)      # Set pin's mode is output
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)

def loop():
    while True:
        hopen()
        time.sleep(3)
        hclose()
        #lift(3,2)
        #stop()
        time.sleep(3)

def destroy():
    GPIO.cleanup()             # 释放数据

if __name__ == '__main__':     # Program start from here
    #setup()
    try:
        loop()
        
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child function destroy() will be  executed.
        destroy()


  
    

