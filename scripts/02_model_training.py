# scripts/02_model_training.py (FINAL UPDATE for SMOTE)
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import joblib
from imblearn.over_sampling import SMOTE # NEW: Import SMOTE

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- 1. Load Feature Matrix ---
df = pd.read_csv('results/feature_matrix.csv', index_col=0)

# --- 2. Prepare Data ---
# Use the top features identified in the last run for a robust model
# Features: N_mut, Fraction_InFrame_SV, Mutation_Position_Variance, Median_VAF
TOP_FEATURES = ['N_mut', 'Fraction_InFrame_SV', 'Mutation_Position_Variance', 'Median_VAF']

# Filter data to only include these top features
X = df[TOP_FEATURES] 
y = df['Is_Driver']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config['TEST_SIZE'], 
    random_state=config['RANDOM_SEED'],
    stratify=y
)

# --- 3. APPLY SMOTE ---
print("\n--- Applying SMOTE to Training Data ---")
smote = SMOTE(sampling_strategy='minority', random_state=config['RANDOM_SEED'])
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original Training Driver count: {y_train.sum()}")
print(f"Resampled Training Driver count: {y_train_resampled.sum()}")
# Calculate new scale_pos_weight for resampled data (should be 1)
scale_pos_weight = (y_train_resampled.value_counts()[0] / y_train_resampled.value_counts()[1])
print(f"New scale_pos_weight: {scale_pos_weight:.2f} (ideally 1.0 for SMOTE)")

# --- 4. Final Model Training on SMOTE'd Data ---
# Use a simple, robust model (max_depth=3)
final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=config['RANDOM_SEED'],
    scale_pos_weight=scale_pos_weight, # Using 1.0 (or near 1.0) now
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    n_jobs=4
)

final_model.fit(X_train_resampled, y_train_resampled)

# --- 5. Final Evaluation and Save ---
# IMPORTANT: Evaluate ONLY on the original, non-SMOTE'd Test Set
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
auprc = average_precision_score(y_test, y_pred_proba)

print(f"\nFinal Model Test Set AUPRC (SMOTE + Selected Features): {auprc:.4f}")

# Save the final robust model and predictions
joblib.dump(final_model, 'results/driver_model_final_smote.pkl') # New name
pd.DataFrame({
    'Gene': X_test.index, 
    'True_Label': y_test, 
    'Prediction_Prob': y_pred_proba
}).to_csv('results/test_predictions_final_smote.csv', index=False)