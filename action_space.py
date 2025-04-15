from robobopy.Robobo import Robobo
import random

def perform_random_action_freely(rob):
    rSpeed = random.randint(-30, 30)
    lSpeed = random.randint(-30, 30)
    rob.moveWheelsByTime(rSpeed, lSpeed, 1, wait=True)
    return rSpeed, lSpeed

def perform_random_action_limited(rob):
    option = random.randint(0,4)
    rSpeed, lSpeed = 0,0
    if option == 0: # 0º
        rSpeed, lSpeed = 30, 30
    elif option == 1: # 45º
        rSpeed, lSpeed = 30, 5
    elif option == 2: # 90º
        rSpeed, lSpeed = 30, 0
    elif option == 3: # -45º
        rSpeed, lSpeed = 5, 30
    else: # -90º
        rSpeed, lSpeed = 0, 30
    rob.moveWheelsByTime(rSpeed, lSpeed, 1, wait=True)
    return rSpeed, lSpeed