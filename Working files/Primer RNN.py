import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Stvaranje sintetičkih podataka za primer
timesteps = 100
data = np.sin(0.1 * np.arange(timesteps)) + np.random.normal(0, 0.1, timesteps)
data = data.reshape(1, timesteps, 1) # Reshape za RNN

# RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(32, input_shape=(timesteps, 1)))
rnn_model.add(Dense(1))

# Kompilacija modela
rnn_model.compile(optimizer='adam', loss='mse')

# Obuka modela
rnn_model.fit(data, data, epochs=100, verbose=0)

# Predviđanje
rnn_predictions = rnn_model.predict(data)