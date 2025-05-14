from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from src.utility_model import train_extrinsic_utility_model
import joblib

## Initialization of the robot
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()

rob.setEmotionTo(Emotions.ANGRY)
rob.moveTiltTo(100,50)

world_scaler = joblib.load("models/xgb_scaler_v7.joblib")
world_model = joblib.load("models/xgb_world_model_v7.joblib")

train_extrinsic_utility_model(sim, rob, 30, world_model, world_scaler, "models/scaler_prueba.joblib", "models/model_prueba.pkl")
 
rob.disconnect()
sim.disconnect()