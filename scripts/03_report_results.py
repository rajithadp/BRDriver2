# scripts/03_report_results.py (Complete Code for Reporting and Plotting)

import pandas as pd
import numpy as np
import yaml
import sys
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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
# Use a simple threshold of 0.5 for binary classification metrics
y_pred = (y_score >= 0.5).astype(int)

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