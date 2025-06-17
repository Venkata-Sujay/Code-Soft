import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the sampled datasets
try:
    train_df = pd.read_csv('fraudTrain.csv')
    test_df = pd.read_csv('fraudTest.csv')
    print("Successfully loaded sampled_train_data.csv and sampled_test_data.csv")
except FileNotFoundError:
    print("Error: Make sure 'sampled_train_data.csv' and 'sampled_test_data.csv' are uploaded.")
    exit()

# Define the target column
target_column = 'is_fraud'

# Separate features (X) and target (y) for both train and test sets
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column] # We have y_test for evaluation in this sampled file

print("\nClass distribution in training data:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\nClass distribution in test data:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)


# --- Feature Engineering and Preprocessing ---

# Drop identifier columns or those unlikely to be useful directly
columns_to_drop = [
    'Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'trans_num',
    'merchant', # Merchant can be high cardinality; dropping for simplicity in this sample
    'job',      # Job can be high cardinality; dropping for simplicity
    'city',     # City can be high cardinality; dropping for simplicity
    'state',    # State might be useful but will be one-hot encoded
]

# Convert date/time columns to datetime objects
X_train['trans_date_trans_time'] = pd.to_datetime(X_train['trans_date_trans_time'])
X_test['trans_date_trans_time'] = pd.to_datetime(X_test['trans_date_trans_time'])
X_train['dob'] = pd.to_datetime(X_train['dob'])
X_test['dob'] = pd.to_datetime(X_test['dob'])

# Extract time-based features
X_train['trans_hour'] = X_train['trans_date_trans_time'].dt.hour
X_test['trans_hour'] = X_test['trans_date_trans_time'].dt.hour
X_train['trans_day_of_week'] = X_train['trans_date_trans_time'].dt.dayofweek
X_test['trans_day_of_week'] = X_test['trans_date_trans_time'].dt.dayofweek
X_train['trans_month'] = X_train['trans_date_trans_time'].dt.month
X_test['trans_month'] = X_test['trans_date_trans_time'].dt.month

# Calculate age from dob (approximate)
current_year = pd.Timestamp.now().year # Using current year, but better to use a fixed reference or last trans year
X_train['age'] = current_year - X_train['dob'].dt.year
X_test['age'] = current_year - X_test['dob'].dt.year

# Drop original date/time and other selected columns
X_train = X_train.drop(columns=columns_to_drop + ['trans_date_trans_time', 'dob'], errors='ignore')
X_test = X_test.drop(columns=columns_to_drop + ['trans_date_trans_time', 'dob'], errors='ignore')

# Identify numerical and categorical features remaining
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

# Define preprocessing steps
# Numerical features are scaled
# Categorical features are one-hot encoded
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Model Training and Evaluation ---

models = {
    "Logistic Regression": Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=2000))]),
    "Decision Tree": Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))]),
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
}

evaluation_results = {}
predictions_df = pd.DataFrame({'Transaction_ID': test_df.index}) # Use DataFrame index as a pseudo-ID

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)

    print(f"--- Evaluating {name} ---")
    y_pred = model.predict(X_test)
    predictions_df[f'Predicted_Fraud_{name.replace(" ", "_")}'] = y_pred

    # Get prediction probabilities for ROC AUC if the model supports it
    y_prob = None
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        print(f"Note: {name} does not support predict_proba.")


    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)']))

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
output_predictions_file = 'all_fraud_predictions.csv'
predictions_df.to_csv(output_predictions_file, index=False)

print(f"\nAll model predictions saved to {output_predictions_file}")
print("\nSample of combined predictions:")
print(predictions_df.head())

print("\n--- Summary of Model Performance ---")
for name, results in evaluation_results.items():
    print(f"\nModel: {name}")
    print(f"  Fraudulent (1) Class - Precision: {results['classification_report']['Fraudulent (1)']['precision']:.4f}")
    print(f"  Fraudulent (1) Class - Recall: {results['classification_report']['Fraudulent (1)']['recall']:.4f}")
    print(f"  Fraudulent (1) Class - F1-Score: {results['classification_report']['Fraudulent (1)']['f1-score']:.4f}")
    if results['roc_auc'] != "N/A":
        print(f"  ROC AUC Score: {results['roc_auc']:.4f}")
    else:
        print("  ROC AUC Score: Not Applicable (due to single class in true labels or predictions)")