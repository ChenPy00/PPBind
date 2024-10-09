# Prepare pre-training data

## download data

zenodo: https://zenodo.org/doi/10.5281/zenodo.11070823
rename the folder as benchmark_data

## split data

cd ./PPI_split

python calculate_interface_embedding.py

## merge data
run ./extra_data/integrate.ipynb


# Updating in progress(PPBind-3D）

......


# Prepare data for fine-tuning

cd ./finetune_data_DIPS-Plus

## Download DIPS-Plus

wget https://zenodo.org/records/5134732/files/final_raw_dips.tar.gz
    
tar -xzf final_raw_dips.tar.gz

## Get Batch PDB Files Downloads with Shell Script

wget https://www.rcsb.org/scripts/batch_download.sh

## process data

python ./get_pair_chain_from_dill.py

This step will generate two files: 

'DIPS-Plus-Pair-Data.csv'

'DIPS-Plus-PDB-Set.txt

## Download PDB from RCSB

mkdir ./DIPS-Plus-PDB

bash ./batch_download.sh -f ./DIPS-Plus-PDB-Set.txt -p -o ./DIPS-Plus-PDB

gunzip *.gz

## Merge pretrain dataset and finetune dataset

python ./finetune_dataset_integrate.py


# Updating in progress(PPBind-1D）

......
