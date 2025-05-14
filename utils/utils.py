import random
import numpy as np
from scipy.spatial.distance import euclidean

def avoid_obstacle(rob, frontal_distance, back_distance):
    rob.stopMotors()
    if frontal_distance > back_distance:
        rob.moveWheelsByTime(-20,-20,2, wait=True)
        rob.moveWheelsByTime(-20,20,1, wait=True)
    else:
        rob.moveWheelsByTime(20,20,2, wait=True)
        rob.moveWheelsByTime(-20,20,1, wait=True)

def get_cylinders_initial_pos(sim, objects):
    cylinders_initial_pos = {
        'REDCYLINDER': None,
        'GREENCYLINDER': None,
        'BLUECYLINDER': None,
        'CUSTOMCYLINDER': None
    }

    if objects != None and len(objects) > 0:
        for object in objects:
            location = sim.getObjectLocation(object)
            if object == 'REDCYLINDER' or object == 'GREENCYLINDER' or object == 'BLUECYLINDER':  # Determine the cylinders position
                cylinders_initial_pos[object] = location['position']
            elif object == 'CUSTOMCYLINDER':  # Move the cylinder to one corner of the simulation
                sim.setObjectLocation('CUSTOMCYLINDER',{'x': -900.0, 'y': 10.0, 'z': 950.0})
    return cylinders_initial_pos


def move_cylinder(sim, cylinder_name):  # Move the cylinder +50 in the X axis (test)
    loc = sim.getObjectLocation(cylinder_name)
    pos = loc['position']
    pos["x"] += random.randint(-150, 150)
    pos["z"] += random.randint(-150, 150)
    sim.setObjectLocation(cylinder_name, pos)

def reset_position_cylinders(sim, cylinders_initial_pos):
    for cylinder_name, initial_position in cylinders_initial_pos.items():
        sim.setObjectLocation(cylinder_name, initial_position)

def get_intrinsinc_utility_from_state(candidate_state, memory_states, n=1.0):
    vector = get_array_from_perceptual_dict(candidate_state)
    if not memory_states:
        return float('inf')  # máxima novedad si no hay memoria
    distances = [euclidean(vector, mem_state) for mem_state in memory_states]
    novelty = np.mean([d**n for d in distances])
    return novelty
    
def objective_found(perceptions: list[dict]):
    distances, indexs = [], []
    for i, perception in enumerate(perceptions):   
        distance_red = perception.get('distance_red')
        angle_red = perception.get('angle_red')
        if (distance_red < 140):
            distances.append(distance_red)
            indexs.append(i)
    if len(distances) > 0:
        return True, indexs[distances.index(min(distances))]
    return False, None

def get_array_from_perceptual_dict(perception: dict):
    return [
        perception.get('distance_red', 0), perception.get('angle_red', 0),
        perception.get('distance_green', 0), perception.get('angle_green', 0),
        perception.get('distance_blue', 0), perception.get('angle_blue', 0)
    ]