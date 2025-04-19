from robobopy.Robobo import Robobo
import random

def avoid_obstacle(rob, frontal_distance, back_distance):
    rob.stopMotors()
    if frontal_distance > back_distance:
        rob.moveWheelsByTime(-20,-20,2, wait=True)
        rob.moveWheelsByTime(-10,10,2, wait=True)
    else:
        rob.moveWheelsByTime(20,20,2, wait=True)
        rob.moveWheelsByTime(-10,10,2, wait=True)

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
        
def save_new_line_of_data(perception_init, rSpeed, lSpeed, perception_final):
    line = f'{perception_init["distance_red"]},{perception_init["angle_red"]},{perception_init["distance_green"]},{perception_init["angle_green"]},'
    line += f'{perception_init["distance_blue"]},{perception_init["angle_blue"]},{rSpeed},{lSpeed},'
    line += f'{perception_final["distance_red"]},{perception_final["angle_red"]},{perception_final["distance_green"]},{perception_final["angle_green"]},'
    line += f'{perception_final["distance_blue"]},{perception_final["angle_blue"]}'
    with open("dataset/dataset_limited.txt", "a") as archivo:
        archivo.write(f"\n{line}")
