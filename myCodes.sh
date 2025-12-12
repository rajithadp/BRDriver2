source ~/.bashrc
sudo apt install snakemake
snakemake --cleanup-metadata .snakemake
rm -rf .snakemake/conda
snakemake --use-conda --cores 4 
# previous step didnt work 
snakemake --use-conda --cores 4 --conda-frontend conda
# for new data
snakemake predict_user_sample --use-conda --cores 4
snakemake results/final_user_predictions.csv --use-conda --cores 4
snakemake results/final_user_predictions.csv --use-conda --cores 4 --latency-wait 60

rm -rf .snakemake/conda
rm -rf .snakemake/metadata
rm -rf .snakemake/log
snakemake results/final_user_predictions.csv --use-conda --cores 4 --latency-wait 60

# this code works
snakemake results/final_user_predictions.csv --use-conda --cores 4 --latency-wait 60 \
--conda-prefix /tmp/snakemake_envs \
--conda-frontend conda

# git hub has a datalimit. so i deleted 
C:\Users\rajit\OneDrive\Documents\repos\breast_cancer_driver_pipeline\data/mutation_file.txt
and 
C:\Users\rajit\OneDrive\Documents\repos\breast_cancer_driver_pipeline\user_data/new_sample_muts.txt

