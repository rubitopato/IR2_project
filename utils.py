from robobopy.Robobo import Robobo

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
                print("si?")
                sim.setObjectLocation('CUSTOMCYLINDER',{'x': -900.0, 'y': 10.0, 'z': 950.0})
                cylinders_initial_pos['CUSTOMCYLINDER'] = {'x': -900.0, 'y': 10.0, 'z': 950.0}
    print(cylinders_initial_pos)
    return cylinders_initial_pos


def move_cylinder(sim, cylinder_name):  # Move the cylinder +50 in the X axis (test)
    loc = sim.getObjectLocation(cylinder_name)
    pos = loc['position']
    pos["x"] += 50
    print("POS: ", pos)
    sim.setObjectLocation(cylinder_name, pos)
    # sim.setObjectLocation("REDCYLINDER", {
    #     'position': {'x': 100.0, 'y': 10.0, 'z': 200.0},
    #     'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
    # })


def reset_position_cylinders(sim, cylinders_initial_pos):
    for cylinder_name, initial_position in cylinders_initial_pos.items():
        sim.setObjectLocation(cylinder_name, initial_position)
