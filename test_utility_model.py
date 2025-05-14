from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
import joblib
from src.utility_model import test_extrinsic_utility_model

## Initialization of the robot
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()

rob.setEmotionTo(Emotions.ANGRY)
rob.moveTiltTo(100,50)

world_scaler = joblib.load("models/xgb_scaler_v7.joblib")
world_model = joblib.load("models/xgb_world_model_v7.joblib")

extrinsic_scaler = joblib.load("models/extrinsic_scaler_definitive.joblib")
extrinsic_model = joblib.load("models/extrinsic_model_definitive.pkl")

test_extrinsic_utility_model(sim, rob, 5, world_model, world_scaler, extrinsic_model, extrinsic_scaler)

rob.disconnect()
sim.disconnect()