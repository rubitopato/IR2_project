from robobopy.Robobo import Robobo
import random
from typing import List, Tuple

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

def generate_candidate_actions(num_actions: int, min_speed: int = -30, max_speed: int = 30) -> list[tuple[int, int]]:
    """
    Generates a list of candidate actions (rSpeed, lSpeed tuples).

    Args:
        num_actions: The number of candidate actions to generate.
        min_speed: The minimum speed for each wheel.
        max_speed: The maximum speed for each wheel.

    Returns:
        A list of tuples, where each tuple is (rSpeed, lSpeed).
    """
    candidate_actions = []
    for _ in range(num_actions):
        rSpeed = random.randint(min_speed, max_speed)
        lSpeed = random.randint(min_speed, max_speed)
        candidate_actions.append((rSpeed, lSpeed))
    # Optional: Add specific useful actions like moving straight or stopping
    # candidate_actions.append((20, 20)) # Move straight
    # candidate_actions.append((0, 0))   # Stop
    return candidate_actions

# --- NEW FUNCTION to generate ALL actions ---
def generate_all_possible_actions(min_speed: int = -30, max_speed: int = 30) -> List[Tuple[int, int]]:
    """
    Generates a list of ALL possible integer action combinations.
    WARNING: This can be a very large list (e.g., 3721 actions for -30 to 30)
             and likely too slow for practical use in the main loop.
    """
    all_actions = []
    for rSpeed in range(min_speed, max_speed + 1):
        for lSpeed in range(min_speed, max_speed + 1):
            all_actions.append((rSpeed, lSpeed))
    print(f"Generated {len(all_actions)} total possible actions.") # Add a print for awareness
    return all_actions