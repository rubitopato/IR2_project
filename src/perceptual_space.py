from robobopy.utils.BlobColor import BlobColor
import math

robot = None
perceptions = {
    BlobColor.RED : [-1, -1, -1, -360],
    BlobColor.GREEN : [-1, -1, -1, -360],
    BlobColor.BLUE : [-1, -1, -1, -360]
}

def get_properties_of_detected_blob():
    if robot.readColorBlob(BlobColor.RED).size != 0:
        if perceptions[BlobColor.RED][2] < robot.readColorBlob(BlobColor.RED).size:
            perceptions[BlobColor.RED] = [
                robot.readColorBlob(BlobColor.RED).posx, robot.readColorBlob(BlobColor.RED).posy, robot.readColorBlob(BlobColor.RED).size, robot.readOrientationSensor().yaw
            ]
    if robot.readColorBlob(BlobColor.GREEN).size != 0:
        if perceptions[BlobColor.GREEN][2] < robot.readColorBlob(BlobColor.GREEN).size:
            perceptions[BlobColor.GREEN] = [
                robot.readColorBlob(BlobColor.GREEN).posx, robot.readColorBlob(BlobColor.GREEN).posy, robot.readColorBlob(BlobColor.GREEN).size, robot.readOrientationSensor().yaw
            ]
    if robot.readColorBlob(BlobColor.BLUE).size != 0:
        if perceptions[BlobColor.BLUE][2] < robot.readColorBlob(BlobColor.BLUE).size:
            perceptions[BlobColor.BLUE] = [
                robot.readColorBlob(BlobColor.BLUE).posx, robot.readColorBlob(BlobColor.BLUE).posy, robot.readColorBlob(BlobColor.BLUE).size, robot.readOrientationSensor().yaw
            ]

def get_perceptual_state(rob):
    global robot
    robot = rob
    perceptions[BlobColor.RED] = [-1,-1,-1, -360]
    perceptions[BlobColor.GREEN] = [-1,-1,-1, -360]
    perceptions[BlobColor.BLUE] = [-1,-1,-1, -360]
    rob.setActiveBlobs(red=True, green=True, blue=True, custom=False)
    rob.moveWheels(10,-10)
    rob.whenANewColorBlobIsDetected(get_properties_of_detected_blob)
    rob.wait(7)
    rob.stopMotors()
    print(perceptions)

def get_distance_and_relative_angle(obj_pos, robot_pos, robot_yaw):
    dx = obj_pos['x'] - robot_pos['x']
    dz = obj_pos['z'] - robot_pos['z']
    distance = math.hypot(dx, dz)
    angle_rad = math.atan2(dz, dx)
    angle_deg = math.degrees(angle_rad)
    angle_relative = angle_deg + robot_yaw - 90
    angle_relative = angle_relative % 360
    return distance, angle_relative    
    
def get_perceptual_state_limited(sim):
    robot_info = sim.getRobotLocation(0)
    robot_yaw = robot_info['rotation']['y']      
    robot_position = sim.getRobotLocation(0)['position']
    red_cylinder_position = sim.getObjectLocation('REDCYLINDER')['position']
    blue_cylinder_position = sim.getObjectLocation('BLUECYLINDER')['position']
    green_cylinder_position = sim.getObjectLocation('GREENCYLINDER')['position']
    
    distance_red, angle_red = get_distance_and_relative_angle(red_cylinder_position, robot_position, robot_yaw)
    distance_blue, angle_blue = get_distance_and_relative_angle(blue_cylinder_position, robot_position, robot_yaw)
    distance_green, angle_green = get_distance_and_relative_angle(green_cylinder_position, robot_position, robot_yaw)

    perceptions_limited = {
        'distance_red': distance_red, 
        'angle_red': angle_red,
        'distance_green': distance_green, 
        'angle_green': angle_green,
        'distance_blue': distance_blue, 
        'angle_blue': angle_blue
    }
    return perceptions_limited
    