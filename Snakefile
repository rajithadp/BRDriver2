# Snakefile (FINAL, DEPLOYMENT-READY VERSION)

import os
import yaml 

# 1. Configuration Setup
configfile: "config/config.yaml" 

# --- GLOBAL VARIABLES FOR PREDICTION WORKFLOW ---
# These are fixed paths for internal and user outputs
USER_FEATURE_MATRIX = "results/temp_user_features.csv" 
FINAL_MODEL_PATH = "results/driver_model_final_smote.pkl"
FINAL_USER_PREDICTIONS = "results/final_user_predictions.csv"

# Ensure the results directory exists
shell("mkdir -p results")

# --------------------------
# Rule: all (Training Workflow Target)
# --------------------------
# Default target for internal training and report generation
rule all:
    input:
        "results/final_report.txt",

# #################################################
# ### A. TRAINING WORKFLOW (Internal Validation) ###
# #################################################

# --------------------------
# Rule: feature_engineering
# --------------------------
rule feature_engineering:
    input:
        sv_file = "data/sv_file.txt",
        mut_file = "data/mutation_file.txt"
    output:
        feature_matrix = "results/feature_matrix.csv"
    conda:
        "envs/ml_env.yaml"
    # Uses shell to pass inputs and output explicitly
    shell:
        """
        python3 scripts/01_feature_engineering.py {input.sv_file} {input.mut_file} {output.feature_matrix}
        """

# --------------------------
# Rule: model_training
# --------------------------
rule model_training:
    input:
        "results/feature_matrix.csv"
    output:
        model = FINAL_MODEL_PATH,
        predictions = "results/test_predictions_final_smote.csv"
    params:
        config_file = "config/config.yaml"
    conda:
        "envs/ml_env.yaml"
    script:
        "scripts/02_model_training.py"

# --------------------------
# Rule: report_results
# --------------------------
rule report_results:
    input:
        predictions = "results/test_predictions_final_smote.csv",
        config = "config/config.yaml"
    output:
        "results/final_report.txt"
    conda:
        "envs/ml_env.yaml"
    shell:
        """
        python3 scripts/03_report_results.py {input.predictions} {output}
        """


# #################################################
# ### B. USER PREDICTION WORKFLOW (External Use) ###
# #################################################

# --------------------------
# Rule: user_feature_engineering (STEP 1: Process User Data)
# --------------------------
rule user_feature_engineering:
    input:
        sv_file = "data/sv_file.txt",
        # Accesses the user-specified file from the loaded config dictionary
        mut_file = config["NEW_MUTATION_FILE"] 
    output:
        temp(USER_FEATURE_MATRIX) # Creates the temp feature matrix
    conda:
        "envs/ml_env.yaml"
    # Uses shell to pass the user input file path
    shell:
        """
        python3 scripts/01_feature_engineering.py {input.sv_file} {input.mut_file} {output}
        """

# --------------------------
# Rule: predict_user_sample (STEP 2: Run Inference)
# --------------------------
# This rule is the final target for external users
rule predict_user_sample:
    input:
        features = USER_FEATURE_MATRIX, # Depends on the output of the previous rule
        model = FINAL_MODEL_PATH
    output:
        prediction_output = FINAL_USER_PREDICTIONS
    conda:
        "envs/ml_env.yaml"
    shell:
        """
        python3 scripts/04_predict_new_data.py {input.features} {input.model} {output.prediction_output}
        """