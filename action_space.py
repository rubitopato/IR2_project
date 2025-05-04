from robobopy.Robobo import Robobo
import random

def generate_random_position(y, existing_positions, min_dist=120):
    while True:
        x = random.randint(-769, 769)
        z = random.randint(-769, 769)
        too_close = any(
            (abs(x - pos["x"])**2 + abs(z - pos["z"])**2)**0.5 < min_dist
            for pos in existing_positions
        )
        if not too_close:
            return {"x": x, "y": y, "z": z}

def reset_randomize_positions(sim, object_names, min_dist=120):
    used_positions = []
    #Reset robot
    robot_loc = sim.getRobotLocation(0)
    robot_pos = robot_loc['position']
    robot_y = robot_pos['y']
    new_pos = generate_random_position(robot_y, used_positions, min_dist=min_dist)
    used_positions.append(new_pos)
    sim.setRobotLocation(0, new_pos, {"x": 0.0, "y": 0.0, "z": 0.0})

    #Reset cylinders
    for obj_name in object_names:
        obj_loc = sim.getObjectLocation(obj_name)
        obj_pos = obj_loc['position']
        obj_y = obj_pos['y']
        new_pos = generate_random_position(obj_y, used_positions, min_dist=min_dist)
        used_positions.append(new_pos)
        sim.setObjectLocation(object_id=obj_name, position=new_pos, rotation={"x": 0.0, "y": 0.0, "z": 0.0})



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