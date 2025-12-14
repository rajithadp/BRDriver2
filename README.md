# 🧬 Breast Cancer Driver Prediction Pipeline (Snakemake)

This pipeline uses machine learning, built with **Snakemake**, to identify cancer driver genes in new patient samples based on genomic alterations (point mutations and structural variants). It leverages robust feature engineering, SMOTE oversampling, and a trained classification model to provide ranked driver predictions. 

## 🌟 Features

* **Custom Feature Engineering:** Calculates features like Mutation Position Variance, Variant Allele Frequency (VAF), Truncating Mutation Fraction, and Mutations per Kilobase (Mut/kb).
* **Structural Variant Integration:** Incorporates data on total structural variants (N_SV) and in-frame vs. out-of-frame fusions.
* **Scalable Workflow:** Built on **Snakemake** for robust, reproducible, and parallel execution.
* **Prediction Mode:** Optimized for predicting driver status on new, unseen patient samples.

---

## 🚀 Getting Started

### Prerequisites

1.  **Conda/Mamba:** You need a working Conda distribution (Miniconda or Mamba) to manage the environments.
2.  **Snakemake:** Install Snakemake globally.

    ```bash
    conda install -c conda-forge snakemake
    ```

### 1. Repository Structure

This is the required directory structure. Ensure your input files and scripts are placed correctly:

```
breast_cancer_driver_pipeline/
├── config/
│     └── config.yaml                # Pipeline parameters, gene lengths, and gold standard list
├── data/
│     ├── mutation_file.txt          # Input training mutations (e.g., TCGA)
│     └── sv_file.txt                # Input training structural variants (e.g., TCGA)
├── envs/
│     └── ml_env.yaml                # Conda environment definition (contains Python, pandas, sklearn, etc.)
├── scripts/
│     ├── 01_feature_engineering.py  # Core feature calculation script
│     └── 04_predict_new_data.py     # Prediction script
├── user_data/
│     └── new_sample_muts.txt        # <--- Placeholder for your new patient data
├── results/                         # Output directory (created automatically)
└── Snakefile                        # The main workflow definition
```

### 2. Prepare User Input Data

Place the MAF-formatted mutation file for the new patient sample you want to analyze inside the `user_data/` directory.

> **Note:** The file **must** be a standard MAF/TSV file (tab-separated) and include the header comments (starting with `#`), which the feature engineering script (`01_feature_engineering.py`) is configured to skip.

---

## ⚙️ Execution

### A. Training and Model Generation (One-time Setup)

Run this command once to build features from the `data/` folder, train the model, and save it to `results/driver_model_final_smote.pkl`.

```bash
snakemake results/final_report.txt --use-conda --cores 4
```
**B. Prediction for New User Samples (Primary Use Case)**
This command executes the full prediction workflow, using the saved model to generate predictions for the sample specified in config/config.yaml.

**⚠️ WSL/Conda Users:** The recommended command below uses specific flags to bypass known Conda/Mamba issues on Windows Subsystem for Linux (WSL) environments.
```
Bash
### RECOMMENDED COMMAND (Use this one!)
snakemake results/final_user_predictions.csv --use-conda --cores 4 --latency-wait 60 \
--conda-prefix /tmp/snakemake_envs \
--conda-frontend conda
```
```
Flag	            Purpose
--use-conda	        Enables environment management via envs/ml_env.yaml.
--latency-wait 60	**Crucial for Networked Drives (WSL/mnt/c):** Prevents read/write conflicts.
--conda-prefix	    **Fixes WSL Errors:** Forces environment creation to a Linux-native directory (/tmp) to avoid "Non-conda folder exists" errors.
--conda-frontend	**Fixes Mamba Errors:** Forces the use of the stable conda command instead of the faster but sometimes problematic mamba.
```

## 📦 Key Configuration
The file config/config.yaml controls the inputs and model parameters. For running the prediction pipeline, ensure the NEW_MUTATION_FILE parameter is set correctly:

YAML
```
config/config.yaml

# ---------------------------------------------
# USER INPUT PARAMETERS for Prediction Workflow
# ---------------------------------------------
NEW_MUTATION_FILE: "user_data/new_sample_muts.txt" # <--- Must point to the user's input file
```

## 📊 Outputs

---

## 📈 Model Performance & Validation

To give users confidence in the model, the training workflow (`snakemake results/final_report.txt`) generates key performance metrics based on the internal validation set.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **ROC AUC** | ~0.95 | Indicates excellent discriminatory power between driver and passenger genes. |
| **Accuracy** | ~0.92 | Overall correctness of predictions. |
| **Recall (Driver)** | ~0.90 | High rate of identifying true positive drivers. |

A graphical representation of the model's ability to distinguish between driver and passenger genes is available after training:



![Receiver Operating Characteristic Curve](results/roc_curve.png)


*(Note: These values are examples. Your actual values will be determined during the training run.)*

The final output is `results/final_user_predictions.csv`, which contains a ranked list of all genes detected in the sample, sorted by the model's predicted driver probability.
```
Column             Description
Gene               Hugo Symbol of the gene.
Prediction_Prob    Model's confidence that the gene is a cancer driver (0.0 to 1.0).
Rank               Rank order, from highest probability (Rank 1) down.
```
```
Example Output
Gene        Prediction_Prob
TP53        0.999407
GATA3       0.999407
CDH1        0.997436
PIK3CA      0.994167
PTEN        0.987136
```


