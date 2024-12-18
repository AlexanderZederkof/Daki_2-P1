import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
file_path = r'C:\\Users\\alexz\\Documents\\AI Programmering\\alzheimers_disease_data.csv' 
alzheimers_data = pd.read_csv(file_path)

# Data preprocessing
cleaned_data = alzheimers_data.dropna()
X = cleaned_data.drop(columns=["PatientID", "DoctorInCharge", "Diagnosis"])
y = cleaned_data["Diagnosis"]

# Apply one-hot encoding for categorical variables or multimodal distributions
X_encoded = pd.get_dummies(X, columns=["Gender", "Ethnicity", "EducationLevel"], drop_first=True)

# Identify categorical and numerical features
categorical_features_encoded = [col for col in X_encoded.columns if any(cat in col for cat in ["Ethnicity", "EducationLevel"])]
numerical_features = [col for col in X_encoded.columns if col not in categorical_features_encoded]

# Apply standard scaling to numerical features only
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

# Separate a hold-out test set
X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42
)

# Define k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform k-fold cross-validation
fold_accuracies = []
print(f"Performing {k}-Fold Cross-Validation for XGBoost")

for train_index, val_index in kf.split(X_temp):
    # Split the data into training and validation sets for this fold
    X_train, X_val = X_temp.iloc[train_index], X_temp.iloc[val_index]
    y_train, y_val = y_temp.iloc[train_index], y_temp.iloc[val_index]

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Validate the model
    y_val_pred = xgb_model.predict(X_val)
    fold_accuracy = accuracy_score(y_val, y_val_pred)
    fold_accuracies.append(fold_accuracy)

# Display cross-validation results
cv_results = {
    "Mean Accuracy": np.mean(fold_accuracies),
    "Standard Deviation": np.std(fold_accuracies)
}
print("\nCross-Validation Results:")
print(f"XGBoost: Mean Accuracy = {cv_results['Mean Accuracy']:.2f}, Std Dev = {cv_results['Standard Deviation']:.2f}")

# Evaluate on the test set
xgb_model.fit(X_temp, y_temp)  # Train on the entire temp set
y_test_pred = xgb_model.predict(X_test)
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]  # For ROC curve

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("XGBoost - Classification Report:\n", classification_report(y_test, y_test_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost')
plt.legend(loc='lower right')
plt.show()


