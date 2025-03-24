import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# Stvaranje sintetičkih podataka
timesteps = 100
data_points = np.sin(0.1 * np.arange(timesteps)) + np.random.normal(0, 0.1, timesteps)

# Reshapovanje podataka za RNN/LSTM (samples, timesteps, features)
data = data_points.reshape((1, timesteps, 1))  # Oblici: (1, 100, 1)

# RNN Model
rnn_model = Sequential([
    SimpleRNN(32, return_sequences=True, input_shape=(timesteps, 1)),  # Dodato return_sequences=True
    Dense(1)
])

# LSTM Model
lstm_model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(timesteps, 1)),  # Dodato return_sequences=True
    Dense(1)
])

# Kompilacija
rnn_model.compile(optimizer='adam', loss='mse')
lstm_model.compile(optimizer='adam', loss='mse')

# Obuka modela (koristimo iste podatke za ulaz i izlaz kao autoencoder)
rnn_model.fit(data, data, epochs=100, verbose=0)
lstm_model.fit(data, data, epochs=100, verbose=0)

# Predviđanje
rnn_predictions = rnn_model.predict(data)
lstm_predictions = lstm_model.predict(data)

# Grafički prikaz
plt.figure(figsize=(12, 6))
plt.plot(data_points, label='Originalni podaci', linestyle='--')
plt.plot(rnn_predictions[0, :, 0], label='RNN Predikcije')
plt.plot(lstm_predictions[0, :, 0], label='LSTM Predikcije')
plt.title('Poređenje RNN i LSTM predikcija')
plt.legend()
plt.show()