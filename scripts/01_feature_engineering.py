# scripts/01_feature_engineering.py (CORRECTED AND COMPLETE)
import pandas as pd
import numpy as np
import yaml
import sys 

# --- 1. Argument Handling and Config Load ---

if len(sys.argv) < 4:
    print("Usage: python 01_feature_engineering.py <sv_input> <mut_input> <output_file>", file=sys.stderr)
    sys.exit(1)

sv_file_path = sys.argv[1] 
mut_file_path = sys.argv[2] 
output_file_path = sys.argv[3] 

# Load config (still needed for GOLD_STANDARD_DRIVERS and GENE_CDS_LENGTHS)
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- 2. Load Data (USING ARGS) ---
sv_df = pd.read_csv(sv_file_path, sep='\t', low_memory=False) 
# FIX: Added comment='#' to handle MAF format header lines
mut_df = pd.read_csv(mut_file_path, sep='\t', low_memory=False, comment='#') 

# --- 3. Initialize Feature Matrix and Gene List ---
all_genes = set(mut_df['Hugo_Symbol'].unique())
all_genes.update(sv_df['Site1_Hugo_Symbol'].unique())
all_genes.update(sv_df['Site2_Hugo_Symbol'].unique())

# FIX: Ensure feature_matrix is defined BEFORE any operations on it
feature_matrix = pd.DataFrame(index=list(all_genes)) 

# --- 4. Process Mutation Data ---
truncating_classes = ['Frame_Shift_Ins', 'Frame_Shift_Del', 'Nonsense_Mutation', 'Splice_Site']

mut_df['is_truncating'] = mut_df['Variant_Classification'].isin(truncating_classes).astype(int)

# --- Aggregation for Count, Impact, and VAF ---
mut_features = mut_df.groupby('Hugo_Symbol').agg(
    N_mut=('Hugo_Symbol', 'size'),
    N_truncating=('is_truncating', 'sum'),
    median_t_depth=('t_depth', 'median'),
    median_t_alt_count=('t_alt_count', 'median'),
    
    # --- Clustering Feature (Variance of Mutation Position) ---
    Mutation_Position_Variance=('Protein_position', lambda x: pd.to_numeric(x, errors='coerce').var(ddof=1))
)

mut_features['Fraction_Truncating'] = mut_features['N_truncating'] / mut_features['N_mut']
mut_features['Median_VAF'] = mut_features['median_t_alt_count'] / mut_features['median_t_depth']
mut_features.replace([np.inf, -np.inf], np.nan, inplace=True)
mut_features.drop(columns=['median_t_depth', 'median_t_alt_count'], inplace=True)

feature_matrix = feature_matrix.merge(mut_features, left_index=True, right_index=True, how='left')


# --- 5. Gene Length Normalization ---
feature_matrix['CDS_Length'] = feature_matrix.index.map(config['GENE_CDS_LENGTHS'])

# Calculate Mutations Per Kilobase (Mutations / length_in_kb)
feature_matrix['Mut_per_kb'] = feature_matrix['N_mut'] / (feature_matrix['CDS_Length'] / 1000)

median_length = feature_matrix['CDS_Length'].median()
# Suppressing inplace warnings for cleanliness
feature_matrix.loc[:, 'CDS_Length'] = feature_matrix['CDS_Length'].fillna(median_length)
feature_matrix.loc[:, 'Mut_per_kb'] = feature_matrix['Mut_per_kb'].fillna(0)


# --- 6. Process Structural Variant (SV) Data ---
sv_gene_counts = {}
inframe_sv_counts = {}
partner_counts = {}

def process_sv_row(row):
    genes = [row['Site1_Hugo_Symbol'], row['Site2_Hugo_Symbol']]
    for gene in genes:
        sv_gene_counts[gene] = sv_gene_counts.get(gene, 0) + 1
        
    if row['Site2_Effect_On_Frame'] == 'in-frame':
        inframe_sv_counts[row['Site2_Hugo_Symbol']] = inframe_sv_counts.get(row['Site2_Hugo_Symbol'], 0) + 1
            
    gene1, gene2 = genes[0], genes[1]
    partner_counts.setdefault(gene1, set()).add(gene2)
    partner_counts.setdefault(gene2, set()).add(gene1)

sv_df.apply(process_sv_row, axis=1)

sv_features = pd.DataFrame({
    'N_SV': pd.Series(sv_gene_counts),
    'N_InFrame_SV': pd.Series(inframe_sv_counts).fillna(0),
    'N_Partners': pd.Series({g: len(p) for g, p in partner_counts.items()})
})

# Handle potential division by zero if N_SV is 0
sv_features['Fraction_InFrame_SV'] = sv_features['N_InFrame_SV'] / sv_features['N_SV'].replace(0, np.nan) 
feature_matrix = feature_matrix.merge(sv_features, left_index=True, right_index=True, how='left')


# --- 7. Final Clean-up and Labeling ---
feature_matrix.fillna(0, inplace=True)
feature_matrix['Is_Driver'] = feature_matrix.index.isin(config['GOLD_STANDARD_DRIVERS']).astype(int)

# --- Save Output (USING ARGS) ---
print(f"Saving feature matrix to {output_file_path}")
feature_matrix.to_csv(output_file_path, index=True)