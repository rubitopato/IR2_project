import random
import time

def generate_random_position(y, existing_positions, min_dist=120):
    while True:
        x = random.randint(-900, 900)
        z = random.randint(-900, 900)
        too_close = any(
            (abs(x - pos["x"])**2 + abs(z - pos["z"])**2)**0.5 < min_dist
            for pos in existing_positions
        )
        if not too_close:
            return {"x": x, "y": y, "z": z}

def reset_randomize_positions(rob, sim, object_names, min_dist=120):
    sim.resetSimulation()
    time.sleep(1)
    used_positions = []

    for obj_name in object_names:
        obj_loc = sim.getObjectLocation(obj_name)
        obj_pos = obj_loc['position']
        obj_y = obj_pos['y']
        new_pos = generate_random_position(obj_y, used_positions, min_dist=min_dist)
        used_positions.append(new_pos)
        sim.setObjectLocation(object_id=obj_name, position=new_pos, rotation={"x": 0.0, "y": 0.0, "z": 0.0})
    
    rob.moveTiltTo(100, 50)