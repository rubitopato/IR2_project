from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from src.world_model import test_world_model

rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()

test_world_model(rob, sim, "models/xgb_world_model_v7.joblib", "models/xgb_scaler_v7.joblib")

rob.disconnect()
sim.disconnect()