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


 # 规定GPIO引脚
IN1 = 18      # 接PUL-
IN2 = 16      # 接PUL+
IN3 = 15      # 接DIR-
IN4 = 13      # 接DIR+
IN5 = 11      # ena+





 
def setStep(w1, w2, w3, w4):
    GPIO.output(IN1, w1)
    GPIO.output(IN2, w2)
    GPIO.output(IN3, w3)
    GPIO.output(IN4, w4)
 
def stop():
    setStep(0, 0, 0, 0)
    GPIO.output(IN5, 1)
 
def downward(delay, steps):
    GPIO.output(IN5, 0)
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
    GPIO.output(IN5, 0)
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
    setup()
    if p1<p2:
        if p1==-1:
            if p2 >= 1:
                p2 = p2 - 1
            upward(0.00005,(p2-p1)*2500-2000)
        else:
            if p1 == 0:
                p2 = p2 - 1
            upward(0.00005,(p2-p1)*2500)
    else:
        if p2==-1:
            if p1 >= 1:
                p1 = p1 - 1
            downward(0.00005,(p1-p2)*2500-2000)
        else:
            if p2 == 0:
                p1 = p1 - 1
            downward(0.00005,(p1-p2)*2500)
    stop() 
  
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
    GPIO.setup(IN5, GPIO.OUT)
    GPIO.output(IN5, 1)


def destroy():
    GPIO.cleanup()             # 释放数据

if __name__ == '__main__':     # Program start from here

    try:
        #lift(4,-1)
        setup()
        stop()
        
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child function destroy() will be  executed.
        destroy()


  
    

