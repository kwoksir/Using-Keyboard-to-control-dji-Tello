import KeyPressModule as kp
from djitellopy import tello
from time import sleep

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
input()
def getKeyboardInput():
    lr, fb, ud, yv = 0,0,0,0
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"):fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"): me.takeoff()
    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])

"""Send RC control via four channels. Command is sent every self.TIME_BTW_RC_CONTROL_COMMANDS seconds.
Arguments:
    left_right_velocity: -100~100 (left/right)
    forward_backward_velocity: -100~100 (forward/backward)
    up_down_velocity: -100~100 (up/down)
    yaw_velocity: -100~100 (yaw)
"""
