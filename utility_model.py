from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import get_array_from_perceptual_dict
import joblib

def get_extrinsic_utility_model():

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),  # arquitectura como en el ejemplo Keras
        activation='relu',
        solver='adam',                 # necesario para usar partial_fit
        learning_rate_init=0.001,
        learning_rate="adaptive",                  # entrenaremos manualmente iteración por iteración            # mantiene el estado entre llamadas a .fit()
    )
    return model

scaler = joblib.load("src/models/extrinsic_scaler3.joblib")

def train_model(model, x, y, fit_scaler):
    x = np.vstack(x)
    y = np.array([item for sublist in y for item in sublist])
    if fit_scaler:
        scaler.fit(x)
        joblib.dump(scaler, "src/models/extrinsic_scaler4.joblib")
        
    x_scaled = scaler.transform(x)
    model.fit(x_scaled, y)
    return model

def get_extrinsic_utility_from_states(predicted_states, model):
    predicted_states_array = [get_array_from_perceptual_dict(predicted_state) for predicted_state in predicted_states]
    predicted_states_numpy_array = np.array(predicted_states_array)
    predicted_states_scaled = scaler.transform(predicted_states_numpy_array)
    utilities = model.predict(predicted_states_scaled)
    return utilities
