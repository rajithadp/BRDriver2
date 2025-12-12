# scripts/04_predict_new_data.py
import pandas as pd
import joblib
import sys
import os

if len(sys.argv) < 4:
    print("Usage: python 04_predict_new_data.py <feature_matrix_input> <model_input> <prediction_output>", file=sys.stderr)
    sys.exit(1)

feature_matrix_file = sys.argv[1]
model_file = sys.argv[2]
prediction_output_file = sys.argv[3]

# 1. Load the Model
print(f"Loading model from: {model_file}")
model = joblib.load(model_file)

# 2. Load the New Feature Matrix
# NOTE: We only need the features the model was trained on!
# The features used were: ['N_mut', 'Fraction_InFrame_SV', 'Mutation_Position_Variance', 'Median_VAF']
try:
    X_new = pd.read_csv(feature_matrix_file, index_col=0)
    
    # Ensure the input data has the exact feature columns the model expects
    required_features = ['N_mut', 'Fraction_InFrame_SV', 'Mutation_Position_Variance', 'Median_VAF']
    X_new = X_new[required_features] 

except KeyError as e:
    print(f"Error: Feature matrix is missing a required column: {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError:
    print(f"Error: Feature matrix file not found at {feature_matrix_file}", file=sys.stderr)
    sys.exit(1)


# 3. Generate Predictions
print(f"Generating predictions for {len(X_new)} genes...")
# We use predict_proba and take the probability of the positive class (index 1)
y_pred_proba = model.predict_proba(X_new)[:, 1]

# 4. Save Results
prediction_results = pd.DataFrame({
    'Gene': X_new.index, 
    'Prediction_Prob': y_pred_proba
}).sort_values(by='Prediction_Prob', ascending=False)

prediction_results.to_csv(prediction_output_file, index=False)
print(f"Prediction results saved to: {prediction_output_file}")
print("\n--- Top Predicted Driver Candidates ---")
print(prediction_results.head(10).to_markdown(index=False))