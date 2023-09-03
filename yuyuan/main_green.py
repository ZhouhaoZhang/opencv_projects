import wheels
import time
import cv
# import step
import hand


def adjust_by_line_mid_a():
    a = cv.detect_line_a()

    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)

    time.sleep(0.1)


def adjust_by_line_mid():
    a = cv.detect_line_a()

    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)

    time.sleep(0.1)

    y = cv.detect_line_y()
    if y >= 0:
        wheels.left(y)
    else:
        wheels.right(-y)
    a = cv.detect_line_a()

    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)

    time.sleep(0.1)


def adjust_by_dot():
    [x, y] = cv.detect_dot_xy()
    if y >= 0:
        wheels.left(y)
    else:
        wheels.right(-y)
    if x >= 0:
        wheels.forward(x)
    else:
        wheels.back(-x)

    time.sleep(0.1)

    a = cv.detect_dot_a()

    if a >= 0:
        wheels.turn_left(a)
    else:
        wheels.turn_right(-a)

    time.sleep(0.1)

    [x, y] = cv.detect_dot_xy()
    if y >= 0:
        wheels.left(y)
    else:
        wheels.right(-y)
    if x >= 0:
        wheels.forward(x)
    else:
        wheels.back(-x)


"""
def adjust_by_circle():

    #（画面视角）
    #爪子与圈中心水平对齐
    #竖直方向差10cm
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

"""


def start_to_line_mid():
    """
    前移130cm
    右移25cm
    对边线调整
    """
    wheels.forward(1300)
    time.sleep(0.1)
    wheels.right(100)
    adjust_by_line_mid()


def line_mid_to_block():
    """
    前移70cm
    调整
    右移10
    前移50
    """
    wheels.forward(700)
    adjust_by_line_mid()
    wheels.right(30)
    adjust_by_line_mid_a()

    wheels.forward(600)


def block_to_line_mid():
    """
    后退60cm
    对边线调整
    后退70cm
    调整
    """
    wheels.back(600)
    adjust_by_line_mid()
    wheels.back(700)
    adjust_by_line_mid()


def line_mid_to_circle():
    """
    退15
    右转90
    前进70
    对点
    前进40
    """
    wheels.back(200)
    adjust_by_line_mid()
    wheels.turn_right(90)
    wheels.forward(550)
    adjust_by_dot()
    wheels.forward(362)


def circle_to_line_mid():
    """
    后退40cm
    对点
    后退70
    左转90
    前进15
    """
    wheels.back(400)
    adjust_by_dot()
    wheels.back(650)
    wheels.turn_left(90)
    wheels.forward(150)
    adjust_by_line_mid()


def circle_to_next_line_mid():
    """
    右转90
    右移25
    退50
    对点
    退70
    左转90
    前进15
    """
    wheels.turn_right(90)
    wheels.right(250)
    wheels.back(500)
    adjust_by_dot()
    wheels.back(700)
    wheels.turn_left(90)
    wheels.forward(150)
    adjust_by_line_mid()


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
    """
    wheels.forward(700)
    adjust_by_line_mid()
    wheels.forward(500)
    wheels.left(250)
    # step.lift(-1, 3)
    # hand.hopenall()


def take():
    [x, y] = cv.detect_block_xy()
    print(x, y)
    if y >= 60:
        if x >= 0:
            wheels.forward(x)
        else:
            wheels.back(-x)
        wheels.left(y)

    else:
        wheels.right(-y + 100)
        if x >= 0:
            wheels.forward(x)
        else:
            wheels.back(-x)
        wheels.left(100)
    time.sleep(0.2)

    hand.hclose()
    time.sleep(0.1)
    # step.lift(-1, 0)


def put(k):
    # step.lift(0, k)
    """"""
    wheels.left(80)
    hand.hopen()
    wheels.right(80)
    # step.lift(k, -1)


def circle_to_next_next_line_mid():
    wheels.turn_right(90)
    wheels.right(250)
    wheels.back(500)
    adjust_by_dot()
    wheels.forward(362)
    wheels.turn_right(90)
    wheels.right(250)
    wheels.back(500)
    adjust_by_dot()
    wheels.back(700)
    wheels.turn_left(90)
    wheels.forward(150)
    adjust_by_line_mid()


if __name__ == "__main__":
    hand.hopen()
    # step.lift(3, -1)
    processing_block = 1
    start_to_line_mid()
    line_mid_to_block()
    take()
    wheels.back(100)
    wheels.right(50)
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_block()
    wheels.left(100)
    take()
    wheels.right(130)
    wheels.forward(50)
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_next_next_line_mid()

    line_mid_to_block()
    take()
    wheels.back(100)
    wheels.right(50)
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_block()
    wheels.left(100)
    take()
    wheels.right(130)
    wheels.forward(50)
    block_to_circle()
    put(processing_block)
    processing_block += 1
    circle_to_next_line_mid()
    line_mid_to_origin()
