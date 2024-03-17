import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = '/home/targol/EpilepticSeizur/data'

# Load the mean PSD features
pre_seizure = np.load(os.path.join(data_dir, 'mean_psd_pre_seizure.npy'))
non_seizure = np.load(os.path.join(data_dir, 'mean_psd_non_seizure.npy'))

# Labels: 1 for pre_seizure, 0 for non_seizure
pre_seizure_labels = np.ones(pre_seizure.shape[0])
non_seizure_labels = np.zeros(non_seizure.shape[0])

# Combine data
X = np.concatenate([pre_seizure, non_seizure], axis=0)
y = np.concatenate([pre_seizure_labels, non_seizure_labels], axis=0)

# Reshape X to be [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode y
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
