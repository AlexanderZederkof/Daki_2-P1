import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Load the dataset.
data2 = pd.read_csv("C:\\Users\\anne\\Desktop\\Daki\\s1\\projekter\\P1\\mri_data_P1\\oasis_longitudinal.csv")

# Data preprocessing.
print(data2.info())

# Encode categorical variables.
data2['Group'] = data2['Group'].replace({'Converted': 1, 'Demented': 2, 'Nondemented': 0})

# Remove irrelevant data.
data2 = data2[data2['Group'] != 2]  # Remove Demented.
data2 = data2[~((data2['Group'] == 1) & (data2['CDR'] > 0.5))]  # Remove Converted with CDR > 0.5.

data2 = data2.drop(['MRI ID', 'Visit', 'Hand', 'CDR'], axis=1)

# Impute missing values with median.
imputer = SimpleImputer(strategy='median')
data2[['SES', 'MMSE']] = imputer.fit_transform(data2[['SES', 'MMSE']])

# Separate features and target. SubjectID is dropped of object.
X = data2.drop(columns=['Group', 'Subject ID'])
y = data2['Group']

# Apply one-hot encoding for categorical variables.
X = pd.get_dummies(X, columns=['M/F'], drop_first=True)

# Apply standard scaling.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance the classes.
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_scaled, y_resampled = smote.fit_resample(X_scaled, y)

# Check class distribution after SMOTE.
print(f"Class distribution after SMOTE: {np.unique(y_resampled, return_counts=True)}")

# Split the data.
X_scaled_temp, X_scaled_test, y_temp, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
X_scaled_train, X_scaled_val, y_train, y_val = train_test_split(X_scaled_temp, y_temp, test_size=0.125, random_state=42)

# Define the XGBoost model.
xgb_model = XGBClassifier(random_state=42, n_estimators=1000, learning_rate=0.1, eval_metric='logloss')

# Set up hyperparameters for GridSearchCV.
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [3, 5, 8, 10],
    "n_estimators": range(100, 500, 1000),
}

# Perform GridSearchCV with Cross-Validation.
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                           scoring='accuracy', n_jobs=-1)

grid_search.fit(X_scaled_train, y_train)

# Get the best parameters and the best model.
print("Best hyperparameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation and test sets.
y_pred = best_model.predict(X_scaled_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")
print(classification_report(y_val, y_pred))

y_test_pred = best_model.predict(X_scaled_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
print(classification_report(y_test, y_test_pred))