import random

def perform_random_action_freely(rob):
    rSpeed = random.randint(-30, 30)
    lSpeed = random.randint(-30, 30)
    rob.moveWheelsByTime(rSpeed, lSpeed, 1, wait=True)
    return rSpeed, lSpeed

def generate_all_possible_actions(min_speed: int = -30, max_speed: int = 30):
    all_actions = []
    for rSpeed in range(min_speed, max_speed + 1):
        for lSpeed in range(min_speed, max_speed + 1):
            all_actions.append((rSpeed, lSpeed))
    return all_actions