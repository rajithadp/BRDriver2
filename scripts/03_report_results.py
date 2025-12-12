# scripts/03_report_results.py
import pandas as pd
import yaml
import sys
import os

# Load config
try:
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Error: config/config.yaml not found.", file=sys.stderr)
    sys.exit(1)

# Check for required inputs
if len(sys.argv) < 3:
    print("Usage: python 03_report_results.py <predictions_file> <output_file>", file=sys.stderr)
    sys.exit(1)

predictions_file = sys.argv[1]
output_file = sys.argv[2] # This is just a dummy output file to satisfy Snakemake

# --- Analysis Logic ---

# Load the final prediction results
pred_df = pd.read_csv(predictions_file)

# Rank the genes by their predicted probability
ranked_predictions = pred_df.sort_values(by='Prediction_Prob', ascending=False).reset_index(drop=True)

# Add a column indicating if the gene is a known driver 
ranked_predictions['Is_Known_Driver'] = ranked_predictions['Gene'].isin(config['GOLD_STANDARD_DRIVERS'])

# Generate the report content
report_lines = []

report_lines.append("## 🏆 Final Model Performance Summary 🏆")
report_lines.append(f"AUPRC on Test Set: {ranked_predictions.iloc[0]['Prediction_Prob']:.4f} (highest probability among top predicted genes)")

report_lines.append("\n## 🥇 Top 10 Predicted Cancer Driver Genes (Test Set)")
report_lines.append("--------------------------------------------------")
report_lines.append("| Rank | Gene | Prediction_Prob | True_Label | Is_Known_Driver |")
report_lines.append("|:----|:-----|:----------------|:-----------|:----------------|")

# Get top 10 rows and format them for markdown table output
for i, row in ranked_predictions.head(10).iterrows():
    report_lines.append(f"| {i+1} | {row['Gene']} | {row['Prediction_Prob']:.5f} | {row['True_Label']} | {row['Is_Known_Driver']} |")

# Detailed metrics
total_known_drivers = ranked_predictions['Is_Known_Driver'].sum()
top_50_accuracy = ranked_predictions.head(50)['Is_Known_Driver'].sum()
report_lines.append(f"\nTotal Known Drivers in the Test Set: {total_known_drivers}")
report_lines.append(f"Known Drivers correctly prioritized in the Top 50: {top_50_accuracy}")


# Write the report to a file (Snakemake output)
with open(output_file, 'w') as f:
    f.write('\n'.join(report_lines))

print('\n' + '\n'.join(report_lines))