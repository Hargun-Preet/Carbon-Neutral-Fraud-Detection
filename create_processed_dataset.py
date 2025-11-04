# AI-Based Carbon-Neutral Fraud Detection
#
# SCRIPT 2: DATA PREPROCESSING
#
# This script is a simplified version of 'run_analysis.py'.
# Its ONLY purpose is to load the two raw 'unprocessed' CSVs,
# run the complete preprocessing pipeline, and save the final,
# 'processed' dataset to a new file.
#
# Run this script ONCE to generate the file for your project submission.

import pandas as pd
import numpy as np
import gc # Garbage Collector
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
print("All libraries imported.")

# --------------------------------------------------
# Step 1: Load and Preprocess the Data (IEEE-CIS)
# --------------------------------------------------
print("\nStep 1: Loading and Preprocessing IEEE-CIS Data...")
print("This may take several minutes and use a lot of RAM.")

try:
    # Load data
    df_train_trans = pd.read_csv('train_transaction.csv')
    df_train_id = pd.read_csv('train_identity.csv')
    
    # Merge
    print("Merging transaction and identity files...")
    df_train = pd.merge(df_train_trans, df_train_id, on='TransactionID', how='left')
    
    del df_train_trans, df_train_id
    gc.collect()

except FileNotFoundError:
    print("\n" + "="*50)
    print("ERROR: Data files not found.")
    print("Please download 'train_transaction.csv' and 'train_identity.csv' from Kaggle.")
    print("Place them in the same folder as this script.")
    print("="*50 + "\n")
    exit()

# --- Preprocessing ---
print("Preprocessing data... This is the longest step.")

# 1. Define feature types
categorical_features = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
    'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
    'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
    'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
]

# 2. Separate Target Variable
y_target = df_train['isFraud']
X_features = df_train.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1) # Drop non-feature cols

# 3. Handle Categorical Features (Label Encoding)
print("Label encoding categorical features...")
for col in X_features.columns:
    if col in categorical_features:
        X_features[col] = X_features[col].astype(str).fillna('__MISSING__')
        le = LabelEncoder()
        X_features[col] = le.fit_transform(X_features[col])
        X_features[col] = X_features[col].astype('int32')

# 4. Handle Numerical Features (Fill NaNs)
print("Filling NaNs in numerical features...")
numerical_cols = [col for col in X_features.columns if col not in categorical_features]
X_features[numerical_cols] = X_features[numerical_cols].fillna(-1)
X_features[numerical_cols] = X_features[numerical_cols].astype('float32')

print(f"\nData preprocessing complete.")
print(f"Full dataset shape: {X_features.shape}")

del df_train
gc.collect()

# --------------------------------------------------
# Step 2: Combine and Save Processed Data
# --------------------------------------------------
print("\nStep 2: Combining features and target variable...")

# Combine the processed features (X) and the target (y) back together
df_processed = pd.concat([X_features, y_target], axis=1)

print(f"Final processed dataframe shape: {df_processed.shape}")

# Save the final, processed dataframe to a new CSV file
# This may take a minute and the file will be large.
output_filename = 'processed_fraud_dataset.csv'
print(f"Saving processed data to '{output_filename}'...")

try:
    df_processed.to_csv(output_filename, index=False)
    print("\n" + "="*50)
    print("SUCCESS!")
    print(f"File '{output_filename}' has been created.")
    print("This is your 'processed' dataset for submission.")
    print("="*50)
except Exception as e:
    print(f"\nAn error occurred while saving the file: {e}")
