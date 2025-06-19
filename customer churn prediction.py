import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load the dataset
try:
    df = pd.read_csv('Churn_Modelling.csv')
    print("Successfully loaded Churn_Modelling.csv")
except FileNotFoundError:
    print("Error: 'Churn_Modelling.csv' not found. Please upload the file.")
    exit()

# Display basic information
print("\nDataset Info:")
print(df.info())

print("\nDataset Head:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

# Identify the target column: 'Exited' is typical for this dataset
target_column = 'Exited'
if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found. Please confirm the name of the churn column.")
    exit()

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"\nClass distribution of '{target_column}':")
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)

# Drop irrelevant columns (e.g., customer ID, names, surname)
# 'RowNumber', 'CustomerId', 'Surname' are typically identifiers and not predictive features.
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
X = X.drop(columns=columns_to_drop, errors='ignore')

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Create a preprocessor for numerical and categorical features
# Numerical features are scaled
# Categorical features are one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Using stratify=y to ensure similar class distribution in train and test sets

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- Model Training and Evaluation ---

models = {
    "Logistic Regression": Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear'))]),
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))]),
    "Gradient Boosting": Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', GradientBoostingClassifier(random_state=42))])
}

evaluation_results = {}
predictions_df = pd.DataFrame({'Original_Index': X_test.index, 'Actual_Churn': y_test})

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)

    print(f"--- Evaluating {name} ---")
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1 = churn)
    else:
        print(f"Note: {name} does not support predict_proba.")

    predictions_df[f'Predicted_Churn_{name.replace(" ", "_")}'] = y_pred
    if y_prob is not None:
        predictions_df[f'Churn_Prob_{name.replace(" ", "_")}'] = y_prob

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Not Churned (0)', 'Churned (1)'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Not Churned (0)', 'Churned (1)']))

    # ROC AUC Score
    roc_auc = "N/A"
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
            print(f"ROC AUC Score: {roc_auc:.4f}")
        except ValueError:
            print("ROC AUC score could not be calculated (e.g., only one class present in y_true or y_score).")

    # Store results
    evaluation_results[name] = {
        "classification_report": report,
        "roc_auc": roc_auc
    }

# Save all predictions to a single CSV file
output_predictions_file = 'churn_predictions_and_probabilities.csv'
predictions_df.to_csv(output_predictions_file, index=False)

print(f"\nAll model predictions and probabilities saved to {output_predictions_file}")
print("\nSample of combined predictions and probabilities:")
print(predictions_df.head())

print("\n--- Summary of Model Performance ---")
for name, results in evaluation_results.items():
    print(f"\nModel: {name}")
    print(f"  Churned (1) Class - Precision: {results['classification_report']['Churned (1)']['precision']:.4f}")
    print(f"  Churned (1) Class - Recall: {results['classification_report']['Churned (1)']['recall']:.4f}")
    print(f"  Churned (1) Class - F1-Score: {results['classification_report']['Churned (1)']['f1-score']:.4f}")
    if results['roc_auc'] != "N/A":
        print(f"  ROC AUC Score: {results['roc_auc']:.4f}")
    else:
        print("  ROC AUC Score: Not Applicable")