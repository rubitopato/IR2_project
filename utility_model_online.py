from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import numpy as np
from utils import get_array_from_perceptual_dict

def get_extrinsic_utility_model():

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),  # arquitectura como en el ejemplo Keras
        activation='relu',
        solver='sgd',                 # necesario para usar partial_fit
        learning_rate_init=0.001,
        max_iter=1,                   # entrenaremos manualmente iteración por iteración
        warm_start=True              # mantiene el estado entre llamadas a .fit()
    )
    return model

scaler = StandardScaler()

def train_model(model, x, y, fit_scaler):
    x = np.array(x)
    y = np.array(y)
    
    if fit_scaler:
        scaler.fit(x)
        
    x_scaled = scaler.transform(x)
    model.partial_fit(x_scaled, y)
    return model

def get_extrinsic_utility_from_states(predicted_states, model):
    predicted_states_array = [get_array_from_perceptual_dict(predicted_state) for predicted_state in predicted_states]
    predicted_states_numpy_array = np.array(predicted_states_array)
    predicted_states_scaled = scaler.transform(predicted_states_numpy_array)
    utilities = model.predict(predicted_states_scaled)
    return utilities
