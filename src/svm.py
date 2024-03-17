import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

data_dir = '/home/targol/EpilepticSeizur/data'

# Load the mean PSD features
mean_psd_pre_seizure = np.load(os.path.join(data_dir, 'mean_psd_pre_seizure.npy'))
mean_psd_non_seizure = np.load(os.path.join(data_dir, 'mean_psd_non_seizure.npy'))

# Create labels
labels_pre_seizure = np.ones(mean_psd_pre_seizure.shape[0])
labels_non_seizure = np.zeros(mean_psd_non_seizure.shape[0])

# Combine data and labels
X = np.concatenate([mean_psd_pre_seizure, mean_psd_non_seizure], axis=0)
y = np.concatenate([labels_pre_seizure, labels_non_seizure], axis=0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print("Random Forest Classifier Results:")
print(classification_report(y_test, rf_predictions))
#
# Train SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
print("SVM Classifier Results:")
print(classification_report(y_test, svm_predictions))

#
# SVM Classifier Results:
#               precision    recall  f1-score   support
#
#          0.0       0.54      0.85      0.66      7819
#          1.0       0.63      0.26      0.37      7720
#
#     accuracy                           0.56     15539
#    macro avg       0.58      0.55      0.51     15539
# weighted avg       0.58      0.56      0.51     15539

# Random Forest Classifier Results:
#               precision    recall  f1-score   support
#
#          0.0       0.88      0.91      0.89      7819
#          1.0       0.91      0.87      0.89      7720
#
#     accuracy                           0.89     15539
#    macro avg       0.89      0.89      0.89     15539
# weighted avg       0.89      0.89      0.89     15539