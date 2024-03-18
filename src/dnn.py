import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

data_dir = '/home/targol/EpilepticSeizur/data'

# Assuming 'data_dir' is already defined
pre_seizure_data = np.load(data_dir + '/pre_seizure_scaled.npy')
non_seizure_data = np.load(data_dir + '/non_seizure_scaled.npy')

# Create labels
pre_seizure_labels = np.ones(pre_seizure_data.shape[0])
non_seizure_labels = np.zeros(non_seizure_data.shape[0])

# Combine the data
X = np.concatenate((pre_seizure_data, non_seizure_data), axis=0)
y = np.concatenate((pre_seizure_labels, non_seizure_labels), axis=0)

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, 2)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the data for DNN
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Dense Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
