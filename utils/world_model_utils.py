from src.perceptual_space import get_perceptual_state_limited
from src.action_space import perform_random_action_freely
from utils.utils import avoid_obstacle
from utils.simulation_utils import reset_randomize_positions
import joblib
import random
from robobopy.utils.IR import IR

def get_world_model_info(sim, rob, data_each_iteration, iteration):
    i = 0
    j = 0
    while i < iteration:
        
        while j < data_each_iteration:
            if rob.readIRSensor(IR.FrontC) > 60 or rob.readIRSensor(IR.BackC) > 85:
                print("Avoiding obstacle")
                avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
                rob.wait(1)
            else:
                perception_init = get_perceptual_state_limited(sim)
                r, l = perform_random_action_freely(rob)
                rob.stopMotors()
                perception_final = get_perceptual_state_limited(sim)
                save_new_line_of_data(perception_init, r, l, perception_final)
                j += 1
                rob.wait(0.1)
                
        objects = sim.getObjects()
        reset_randomize_positions(sim, objects, 250)
        rob.wait(1)    
        i += 1
        j = 0
        
def load_world_model(world_model_path: str):
    return joblib.load(world_model_path)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

def get_possible_action(under_limit: int, upper_limit: int):
    rSpeed = random.randint(under_limit, upper_limit)
    lSpeed = random.randint(under_limit, upper_limit)
    return rSpeed, lSpeed

def save_new_line_of_data(perception_init, rSpeed, lSpeed, perception_final):
    line = f'{perception_init["distance_red"]},{perception_init["angle_red"]},{perception_init["distance_green"]},{perception_init["angle_green"]},'
    line += f'{perception_init["distance_blue"]},{perception_init["angle_blue"]},{rSpeed},{lSpeed},'
    line += f'{perception_final["distance_red"]},{perception_final["angle_red"]},{perception_final["distance_green"]},{perception_final["angle_green"]},'
    line += f'{perception_final["distance_blue"]},{perception_final["angle_blue"]}'
    with open("dataset/new_dataset3.txt", "a") as archivo:
        archivo.write(f"\n{line}")