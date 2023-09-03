import wheels
import time
import cv
import step
import hand


def adjust_by_line_mid():
    a = cv.detect_line_a()
    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)
    time.sleep(0.5)
    y = cv.detect_line_y()
    if y >= 0:
        wheels.left(y)
    else:
        wheels.right(-y)


def adjust_by_circle():
    """
    （画面视角）
    爪子与圈中心水平对齐
    竖直方向差10cm
    """
    a = cv.detect_circle_a()
    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)
    time.sleep(0.5)
    [x, y] = cv.detect_circle_xy()
    if y >= 0:
        wheels.left(y)
    else:
        wheels.right(-y)
    time.sleep(0.5)
    if x >= 0:
        wheels.forward(x)
    else:
        wheels.back(-x)


def start_to_line_mid():
    """
    右移35cm
    前移135cm
    对边线调整
    """
    wheels.right(350)
    time.sleep(0.5)
    wheels.forward(1350)
    adjust_by_line_mid()


def line_mid_to_block():
    """
    前移120cm
    """
    wheels.forward(1200)


def block_to_line_mid():
    """
    右移10cm
    后退115cm
    对边线调整
    逆时针90度
    """
    wheels.right(100)
    time.sleep(0.5)
    wheels.back(1150)
    adjust_by_line_mid()
    wheels.turn_left(90)
    time.sleep(0.5)


def line_mid_to_circle():
    """
    前进170cm
    对圈调整
    """
    wheels.forward(1700)
    adjust_by_circle()


def circle_to_line_mid():
    """
    后退170cm
    顺时针90度
    对边线调整
    """
    wheels.back(1700)
    time.sleep(0.5)
    wheels.left(100)
    time.sleep(0.5)
    wheels.turn_right(90)
    adjust_by_line_mid()


def circle_to_next_line_mid():
    wheels.turn_left(90)
    time.sleep(0.5)
    circle_to_line_mid()
    wheels.forward(100)


def block_to_circle():
    """
    块到线中
    """
    block_to_line_mid()
    line_mid_to_circle()


def block_to_circle_45():
    pass


def circle_to_block():
    """
    圈到线中
    线中到块
    """
    circle_to_line_mid()
    line_mid_to_block()


def circle_to_block_45():
    pass


def line_mid_to_origin():
    """
    前移135
    左移35
    """
    wheels.forward(1350)
    time.sleep(0.5)
    wheels.left(350)


def take():
    [x, y] = cv.detect_block_xy()

    if y >= 0:
        if x >= 0:
            wheels.forward(x)
        else:
            wheels.back(-x)
        wheels.left(y)

    else:
        wheels.back(-y + 150)
        if x >= 0:
            wheels.forward(x)
        else:
            wheels.back(-x)
        wheels.left(150)

    hand.hclose()
    step.lift(-1, 0)


def put(k):
    step.lift(0, k)
    wheels.left(100)
    hand.hopen()
    wheels.right(100)
    step.lift(k, -1)


processing_block = 1
start_to_line_mid()

for i in range(3):
    line_mid_to_block()
    wheels.left(100)
    take()
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_block()
    wheels.forward(150)
    wheels.left(250)
    take()
    wheels.right(150)
    wheels.back(150)
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_next_line_mid()

line_mid_to_origin()
