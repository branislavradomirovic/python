import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model

# 1. Priprema podataka
# Pretpostavimo da imamo EEG podatke u obliku numpy niza (n_samples, n_timesteps, n_channels)
# n_samples - broj uzoraka
# n_timesteps - broj vremenskih koraka u svakom uzorku
# n_channels - broj EEG kanala

# Primjer:
n_samples = 100
n_timesteps = 50
n_channels = 10
eeg_data = np.random.rand(n_samples, n_timesteps, n_channels)
eeg_labels = np.random.randint(0, 2, n_samples)  # 0 - normalno, 1 - abnormalno

# 2. Izgradnja LSTM modela
model = Sequential()
model.add(LSTM(64, input_shape=(n_timesteps, n_channels), return_sequences=True))
model.add(Dropout(0.2))  # Dodavanje Dropout sloja za sprečavanje pretreniranja
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Izlazni sloj s sigmoidnom funkcijom za binarnu klasifikaciju

# 3. Vizualizacija arhitekture modela
plot_model(model, to_file='lstm_architecture.png', show_shapes=True, show_dtype=True)

# 4. Kompilacija modela
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Obuka modela
model.fit(eeg_data, eeg_labels, epochs=10, batch_size=32)

# 6. Evaluacija modela
loss, accuracy = model.evaluate(eeg_data, eeg_labels)
print('Loss:', loss)
print('Accuracy:', accuracy)

# 7. Predviđanje
predictions = model.predict(eeg_data)