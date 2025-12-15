# scripts/03_report_results.py (Complete Code for Reporting and Plotting)

import pandas as pd
import numpy as np
import yaml
import sys
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# --- 1. Argument Handling ---

# Requires 3 arguments: predictions_file, report_output_file, plot_output_file
if len(sys.argv) < 4:
    print("Usage: python 03_report_results.py <predictions_csv> <report_output_txt> <plot_output_png>", file=sys.stderr)
    sys.exit(1)

predictions_file = sys.argv[1]
report_output_file = sys.argv[2]
plot_output_file = sys.argv[3]

# --- 2. Load Data ---
try:
    pred_df = pd.read_csv(predictions_file, index_col=0)
except FileNotFoundError:
    print(f"Error: Predictions file not found at {predictions_file}", file=sys.stderr)
    sys.exit(1)

# Check if required columns are present
if 'True_Label' not in pred_df.columns or 'Prediction_Prob' not in pred_df.columns:
    print("Error: Predictions CSV must contain 'True_Label' and 'Prediction_Prob' columns.", file=sys.stderr)
    sys.exit(1)

# --- 3. Calculate Metrics ---

y_true = pred_df['True_Label']
y_score = pred_df['Prediction_Prob']

# --- Find optimal threshold for imbalanced data ---
# Method 1: Try to find threshold with good precision
test_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

best_precision = 0
best_threshold = 0.5
best_y_pred = None

for threshold in test_thresholds:
    y_pred_test = (y_score >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_test).ravel()
    
    # Calculate precision
    if tp + fp > 0:
        precision_test = tp / (tp + fp)
    else:
        precision_test = 0
    
    # Calculate recall
    if tp + fn > 0:
        recall_test = tp / (tp + fn)
    else:
        recall_test = 0
    
    print(f"Testing threshold {threshold}: Precision={precision_test:.3f}, Recall={recall_test:.3f}")
    
    # Choose threshold with best precision (and at least 1 TP found)
    if precision_test > best_precision and tp > 0:
        best_precision = precision_test
        best_threshold = threshold
        best_y_pred = y_pred_test

print(f"\nSelected threshold: {best_threshold} (Precision: {best_precision:.3f})")

# Use the best threshold found
y_pred = best_y_pred if best_y_pred is not None else (y_score >= 0.5).astype(int)

# ROC AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Classification Report
report = classification_report(y_true, y_pred, target_names=['Passenger', 'Driver'], output_dict=True)

# --- 4. Generate Text Report (final_report.txt) ---

with open(report_output_file, 'w') as f:
    f.write("## Model Performance Report\n")
    f.write(f"--- \n")
    f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
    f.write(f"Accuracy: {report['accuracy']:.4f}\n")
    f.write(f"Driver Recall (Sensitivity): {report['Driver']['recall']:.4f}\n")
    f.write(f"Passenger Specificity: {report['Passenger']['recall']:.4f}\n\n")

    f.write("### Full Classification Report\n")
    # Using pandas to format the report nicely for the text file
    report_df = pd.DataFrame(report).transpose()
    f.write(report_df.to_markdown(floatfmt=".4f"))
    f.write("\n\n")

    f.write("### Confusion Matrix\n")
    cm = confusion_matrix(y_true, y_pred)
    f.write(f"[[TN, FP],\n [FN, TP]]\n")
    f.write(str(cm) + '\n')


# --- 5. Generate ROC Curve Plot (roc_curve.png) ---

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot to the specified output path
plt.savefig(plot_output_file)
plt.close()

print(f"Report saved to {report_output_file}")
print(f"ROC plot saved to {plot_output_file}")