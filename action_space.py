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

def generate_all_possible_actions(min_speed: int = -30, max_speed: int = 30):
    """
    Generates a list of ALL possible integer action combinations.
    WARNING: This can be a very large list (e.g., 3721 actions for -30 to 30)
             and likely too slow for practical use in the main loop.
    """
    all_actions = []
    for rSpeed in range(min_speed, max_speed + 1):
        for lSpeed in range(min_speed, max_speed + 1):
            all_actions.append((rSpeed, lSpeed))
    #print(f"Generated {len(all_actions)} total possible actions.")
    return all_actions