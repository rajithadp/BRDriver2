# scripts/02_model_training.py - FIXED for Severe Imbalance
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import joblib
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- 1. Load Feature Matrix ---
df = pd.read_csv('results/feature_matrix.csv', index_col=0)

# --- 2. Prepare Data ---
EXISTING_FEATURES = [
    'N_mut',
    'Mut_per_kb',           # This exists!
    'Median_VAF',
    'Fraction_InFrame_SV',
    'Mutation_Position_Variance',
    'Fraction_Truncating',
    'N_Partners'
]

# Filter to features that actually exist
TOP_FEATURES = [f for f in EXISTING_FEATURES if f in df.columns]
print(f"Using {len(TOP_FEATURES)} existing features: {TOP_FEATURES}")

X = df[TOP_FEATURES]
y = df['Is_Driver'].values

print(f"Total dataset: {len(X)} genes")
print(f"  - Drivers: {y.sum()} genes ({y.sum()/len(y)*100:.2f}%)")
print(f"  - Passengers: {len(y) - y.sum()} genes")

# --- CRITICAL: Check if we have enough drivers for CV ---
if y.sum() < 10:
    print("\n⚠️  WARNING: Very few drivers detected!")
    print(f"   Only {y.sum()} driver genes found.")
    print("   Consider: Collecting more labeled data or using transfer learning")

# --- 3. Modified Cross-Validation for Severe Imbalance ---
skf = StratifiedKFold(n_splits=min(3, y.sum()), shuffle=True, random_state=config['RANDOM_SEED'])
all_fold_predictions = []

print(f"\n--- Starting {skf.n_splits}-Fold Cross-Validation ---")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\n  Fold {fold}:")
    print(f"    Train: {len(y_train)} total, {y_train.sum()} drivers")
    print(f"    Test: {len(y_test)} total, {y_test.sum()} drivers")
    
    # --- FIX: Adaptive SMOTE based on available drivers ---
    n_drivers_in_train = y_train.sum()
    
    if n_drivers_in_train >= 5:  # Enough drivers for SMOTE
        # Use SMOTE only if we have enough drivers
        try:
            smote_ratio = min(0.1, (5 / len(y_train)))  # Cap at 10% or 5 drivers
            smote = SMOTE(
                sampling_strategy=smote_ratio,
                random_state=config['RANDOM_SEED'] + fold,
                k_neighbors=min(5, n_drivers_in_train - 1)  # Adaptive k
            )
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"    Applied SMOTE: {len(y_train_res)} samples")
        except:
            # Fallback if SMOTE fails
            X_train_res, y_train_res = X_train, y_train
            print(f"    SMOTE failed, using original data")
    else:
        # Not enough drivers for SMOTE - use original data
        X_train_res, y_train_res = X_train, y_train
        print(f"    Not enough drivers for SMOTE (need ≥5, have {n_drivers_in_train})")
    
    # Calculate weight based on ORIGINAL imbalance
    if y_train.sum() > 0:
        fold_scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        fold_scale_pos_weight = min(fold_scale_pos_weight, 100)  # Cap at 100
    else:
        fold_scale_pos_weight = 10  # Default if no drivers in train
    
    # --- Train model ---
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=config['RANDOM_SEED'] + fold,
        scale_pos_weight=fold_scale_pos_weight,
        max_depth=3,  # Keep shallow to prevent overfitting
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=2
    )
    
    model.fit(X_train_res, y_train_res)
    
    # Predict on test fold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    if y_test.sum() > 0:  # Only calculate AUPRC if we have drivers in test
        fold_auprc = average_precision_score(y_test, y_pred_proba)
        print(f"    Test AUPRC = {fold_auprc:.4f}")
    else:
        fold_auprc = np.nan
        print(f"    No drivers in test set")
    
    # Store predictions
    fold_df = pd.DataFrame({
        'Gene': X_test.index,
        'True_Label': y_test,
        'Prediction_Prob': y_pred_proba,
        'Fold': fold
    })
    all_fold_predictions.append(fold_df)

# --- 4. Combine Results ---
if all_fold_predictions:
    cv_predictions = pd.concat(all_fold_predictions, ignore_index=True)
    
    if cv_predictions['True_Label'].sum() > 0:
        cv_auprc = average_precision_score(
            cv_predictions['True_Label'], 
            cv_predictions['Prediction_Prob']
        )
        print(f"\n✅ Cross-Validation AUPRC: {cv_auprc:.4f}")
        print(f"   Evaluated on {cv_predictions['True_Label'].sum()} total driver instances")
    else:
        print("\n⚠️  No drivers found in any test fold!")
        cv_auprc = np.nan
    
    cv_predictions.to_csv('results/test_predictions_cv.csv', index=False)
else:
    print("\n❌ No predictions generated!")
    cv_predictions = pd.DataFrame()
    cv_auprc = np.nan

# --- 5. Train Final Model on ALL Data ---
print("\n--- Training Final Model on All Data ---")

# Use ADASYN instead of SMOTE for final model (more robust for tiny classes)
try:
    from imblearn.over_sampling import ADASYN
    
    # Check if we have enough drivers for any oversampling
    if y.sum() >= 5:
        adasyn = ADASYN(
            sampling_strategy=0.05,  # Target 5% drivers
            random_state=config['RANDOM_SEED'],
            n_neighbors=min(4, y.sum() - 1)
        )
        X_res, y_res = adasyn.fit_resample(X, y)
        print(f"Applied ADASYN: {len(y_res)} total samples")
    else:
        # Not enough drivers - use weighted learning only
        X_res, y_res = X, y
        print(f"Not enough drivers for ADASYN (need ≥5, have {y.sum()})")
except:
    # Fallback to SMOTE with safe parameters
    if y.sum() >= 6:
        smote = SMOTE(
            sampling_strategy=0.05,
            random_state=config['RANDOM_SEED'],
            k_neighbors=min(5, y.sum() - 1)
        )
        X_res, y_res = smote.fit_resample(X, y)
        print(f"Applied SMOTE: {len(y_res)} total samples")
    else:
        X_res, y_res = X, y
        print(f"Using original data (only {y.sum()} drivers)")

# Final model with strong weighting
if y.sum() > 0:
    final_scale_pos_weight = min(100, (len(y) - y.sum()) / y.sum())
else:
    final_scale_pos_weight = 50  # Strong penalty if no drivers

final_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=config['RANDOM_SEED'],
    scale_pos_weight=final_scale_pos_weight,
    max_depth=3,
    n_estimators=150,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=4
)

final_model.fit(X_res, y_res)

# === ADD THIS SECTION at the end, before saving ===
print("\n" + "="*60)
print("MODEL INTERPRETATION & BIOLOGICAL VALIDATION")
print("="*60)

# 1. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': TOP_FEATURES,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n1. FEATURE IMPORTANCE:")
print(feature_importance.to_string())

# 2. Predictions for known drivers
print("\n2. PREDICTIONS FOR KNOWN DRIVER GENES:")
known_drivers = df[df['Is_Driver'] == 1].index.tolist()

for gene in known_drivers:
    if gene in X.index:
        features = X.loc[gene]
        prob = final_model.predict_proba([features])[0, 1]
        is_predicted = prob >= 0.6  # Your optimal threshold
        
        print(f"\n{gene}:")
        print(f"  Prediction Probability: {prob:.3f} → {'DRIVER' if is_predicted else 'NOT PREDICTED'}")
        print(f"  Features:")
        for feat in TOP_FEATURES:
            print(f"    - {feat}: {features[feat]:.3f}")

# 3. Top predicted novel candidates
print("\n3. TOP PREDICTED NOVEL CANDIDATES (Non-driver genes):")
non_drivers = df[df['Is_Driver'] == 0]

# Get predictions for all non-driver genes
novel_candidates = []
for idx, row in non_drivers.iterrows():
    if idx in X.index:
        features = X.loc[idx]
        prob = final_model.predict_proba([features])[0, 1]
        if prob >= 0.6:  # Above your precision threshold
            novel_candidates.append({
                'Gene': idx,
                'Probability': prob,
                'N_mut': features['N_mut'],
                'Features': features.to_dict()
            })

if novel_candidates:
    novel_df = pd.DataFrame(novel_candidates).sort_values('Probability', ascending=False)
    print(f"Found {len(novel_candidates)} novel candidate drivers:")
    print(novel_df[['Gene', 'Probability', 'N_mut']].head(10).to_string())
else:
    print("No novel candidates above threshold 0.6")

# --- 6. Save Everything ---
joblib.dump(final_model, 'results/driver_model_final_imbalance.pkl')

print(f"\n📊 FINAL STATS:")
print(f"   Total genes: {len(X)}")
print(f"   Driver genes: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"   Final scale_pos_weight: {final_scale_pos_weight:.1f}")
if not np.isnan(cv_auprc):
    print(f"   CV AUPRC: {cv_auprc:.4f}")

print("\n✅ Model saved as: results/driver_model_final_imbalance.pkl")
if not cv_predictions.empty:
    print("✅ CV predictions: results/test_predictions_cv.csv")