import cv
import wheels
import hand


def adjust_by_line():
    a = cv.detect_line_a()
    if a >= 0:
        wheels.turn_right(a)
    else:
        wheels.turn_left(-a)
    x = cv.detect_line_x()
    if x >= 0:
        wheels.right(x)
    else:
        wheels.left(-x)


def adjust_by_block():
    a = cv.detect_block_a()
    if a >= 0:
        wheels.turn_right(a)
    else:
        wheels.turn_left(-a)
    x, y = cv.detect_block_xy()
    if x >= 0:
        wheels.back(x)
    else:
        wheels.foward(x)
    wheels.right(y)


def adjust_by_code():
    a = cv.detect_code_a()
    if a >= 0:
        wheels.turn_right(a)
    else:
        wheels.turn_left(-a)
    x, y = cv.detect_code_xy()
    if x >= 0:
        wheels.back(x)
    else:
        wheels.foward(x)
    wheels.right(y)


def take():
    hand.hclose()
    hand.lift(-1, 0)


def start_to_block1():
    """
    起点到第一个块
    """
    wheels.foward(1200)
    adjust_by_line()
    wheels.foward(1000)
    wheels.turn_left(90)


def block1_to_center():
    """
    第一个块到中心
    """
    adjust_by_block()
    take()
    wheels.left(100)
    wheels.turn_left(90)
    wheels.foward(1000)
    wheels.right(500)
    adjust_by_code()
    wheels.right(500)
    

def center_to_block2():
    """
    中心到第二个块
    :return:
    """


def block2_to_center():
    """
    第二个块到中心
    :return:
    """


def center_to_block0():
    """
    中心到块（两个块都已经没了）
    :return:
    """


def to_start():
    """
    返回起点
    :return:
    """


for i in range(3):
    start_to_block1()
    block1_to_center()
    center_to_block2()
    block2_to_center()
    center_to_block0()
