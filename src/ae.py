import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel

data_dir = '/home/targol/EpilepticSeizur/data'

# Load the mean PSD features
X_pre_seizure = np.load(os.path.join(data_dir, 'mean_psd_pre_seizure.npy'))
X_non_seizure = np.load(os.path.join(data_dir, 'mean_psd_non_seizure.npy'))
y_pre_seizure = np.ones(X_pre_seizure.shape[0])
y_non_seizure = np.zeros(X_non_seizure.shape[0])

X = np.concatenate([X_pre_seizure, X_non_seizure], axis=0)
y = np.concatenate([y_pre_seizure, y_non_seizure])

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Define the hypermodel
class ClassifierHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                        input_shape=(self.input_shape,)))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model


hypermodel = ClassifierHyperModel(input_shape=X_train.shape[1])

# Hyperparameter Tuning
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='hparam_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2,
             callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model
y_pred = best_model.predict(X_test) > 0.5
print(classification_report(y_test, y_pred))
