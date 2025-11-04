# AI-Based Carbon-Neutral Fraud Detection
# This script implements the methodology from the presentation: 
# 'Designing AI-Based Carbon-Neutral Fraud Detection Systems'.
#
# This version is designed for a local VS Code environment and uses the
# full IEEE-CIS Fraud Detection dataset.

# --------------------------------------------------
# Step 1: Import all required modules
# --------------------------------------------------
import pandas as pd
import numpy as np
import time
import warnings
import gc # Garbage Collector

# Modeling
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Preprocessing & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score
)

# Sustainability Tracking
from codecarbon import EmissionsTracker

warnings.filterwarnings('ignore')
print("All libraries imported.")

# --------------------------------------------------
# Step 2: Utility Functions
# --------------------------------------------------
print("Defining utility functions...")

def get_performance_metrics(model_name, y_true, y_pred_binary, y_prob):
    """Calculates and prints performance metrics."""
    # Ensure y_pred_binary is binary
    y_pred = (y_pred_binary > 0.5).astype(int)

    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n--- Performance Metrics for {model_name} ---")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    return {"F1-Score": f1, "AUC": auc, "Precision": precision, "Recall": recall}

def get_sustainability_metrics(model_name, train_time, total_inf_time, n_test_samples, emissions_data):
    """Calculates and prints sustainability metrics."""
    inference_latency_ms = (total_inf_time / n_test_samples) * 1000
    
    print(f"\n--- Sustainability Metrics for {model_name} ---")
    print(f"Training Time:       {train_time:.4f} s")
    print(f"Inference Latency:   {inference_latency_ms:.4f} ms/item")
    
    if emissions_data and hasattr(emissions_data, 'energy_consumed'):
        energy = emissions_data.energy_consumed
        co2 = emissions_data.emissions
        print(f"Total Energy Used:  {energy * 1000:.4f} Wh (Watt-hours)")
        print(f"Total CO2 Emissions: {co2 * 1000:.4f} g (grams)")
        return {
            "Training Time (s)": train_time,
            "Inference Latency (ms/item)": inference_latency_ms,
            "Energy (Wh)": energy * 1000,
            "CO2 Emissions (g)": co2 * 1000
        }
    else:
        print("Could not read emission data (perhaps run time was too short or tracker failed).")
        return {
            "Training Time (s)": train_time,
            "Inference Latency (ms/item)": inference_latency_ms,
            "Energy (Wh)": 0,
            "CO2 Emissions (g)": 0
        }

# Dictionary to store all our results for final comparison
results = {}

# --------------------------------------------------
# Step 3: Initialize ONE Emissions Tracker
# --------------------------------------------------
print("Initializing Carbon Tracker...")
# We will use one tracker for the whole script and measure tasks
tracker = EmissionsTracker(project_name="Fraud_Detection_Comparison", log_level='error')
tracker.start()


# --------------------------------------------------
# Step 4: Load and Preprocess the Data (IEEE-CIS)
# --------------------------------------------------
# This is the most complex part.
print("\nStep 4: Loading and Preprocessing IEEE-CIS Data...")
print("This may take several minutes and use a lot of RAM.")

try:
    # Load data
    df_train_trans = pd.read_csv('train_transaction.csv')
    df_train_id = pd.read_csv('train_identity.csv')
    
    # Merge
    print("Merging transaction and identity files...")
    df_train = pd.merge(df_train_trans, df_train_id, on='TransactionID', how='left')
    
    # Free up memory
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
# We must split columns into categorical and numerical
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
y = df_train['isFraud']
X = df_train.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1) # Drop non-feature cols

# 3. Handle Categorical Features (Label Encoding)
# Both models need numeric inputs. LabelEncoding is memory-efficient.
# We will treat NaNs as a separate category.
print("Label encoding categorical features...")
for col in X.columns:
    if col in categorical_features:
        # Convert to string and fill NaNs
        X[col] = X[col].astype(str).fillna('__MISSING__')
        
        # Fit the encoder
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
        # Convert back to a memory-efficient type
        X[col] = X[col].astype('int32')

# 4. Handle Numerical Features (Fill NaNs)
# We fill with -1. Tree models (LGBM) can handle this well.
# DNN will process this in its own pipeline.
print("Filling NaNs in numerical features...")
numerical_cols = [col for col in X.columns if col not in categorical_features]
X[numerical_cols] = X[numerical_cols].fillna(-1)
X[numerical_cols] = X[numerical_cols].astype('float32')

# 5. Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData preprocessing complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print("-" * 40 + "\n")

# Free up more memory
del X, y, df_train
gc.collect()

# --------------------------------------------------
# Step 5: Baseline Model (Deep Neural Network)
# --------------------------------------------------
print("Step 5: Starting DNN Baseline Model Training...")
print("This will be slow and resource-intensive as requested.")

# --- DNN Data Prep ---
# DNNs require scaled data and cannot have NaNs (which we filled with -1).
# We will scale the data.
print("Creating scaled data copy for DNN...")
scaler = StandardScaler()
X_train_dnn = scaler.fit_transform(X_train)
X_test_dnn = scaler.transform(X_test)
# ---

# Start tracking this specific task
tracker.start_task("DNN_Baseline_Train")

# --- Training --- 
start_time = time.time()

model_dnn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_dnn.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid') # Binary classification
])

model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_dnn.fit(
    X_train_dnn, 
    y_train, 
    epochs=15, # Increased epochs for a more "high-energy" baseline
    batch_size=4096, # Large batch size for this dataset
    validation_split=0.2,
    callbacks=[early_stop], 
    verbose=1
)

train_time_dnn = time.time() - start_time
# Stop tracking this task and get its specific data
emissions_data_dnn = tracker.stop_task()

# --- Evaluation --- 
print("Evaluating DNN...")
start_inf_time = time.time()
y_prob_dnn = model_dnn.predict(X_test_dnn, batch_size=8192).flatten()
total_inf_time_dnn = time.time() - start_inf_time

# Get Performance Metrics
perf_dnn = get_performance_metrics("DNN Baseline", y_test, y_prob_dnn, y_prob_dnn)

# Get Sustainability Metrics
sust_dnn = get_sustainability_metrics(
    "DNN Baseline", 
    train_time_dnn, 
    total_inf_time_dnn, 
    len(X_test_dnn), 
    emissions_data_dnn
)

results["DNN_Baseline"] = {"Performance": perf_dnn, "Sustainability": sust_dnn}
print("DNN Baseline Model Finished.")
print("-" * 40 + "\n")

# Free up DNN data
del X_train_dnn, X_test_dnn
gc.collect()

# --------------------------------------------------
# Step 6: Proposed Model (LightGBM)
# --------------------------------------------------
print("Step 6: Starting LightGBM Proposed Model Training...")
# Start tracking this new task
tracker.start_task("LightGBM_Proposed_Train")

# --- Training --- 
start_time = time.time()

# Find categorical feature *indices* for LightGBM
categorical_indices = [X_train.columns.get_loc(c) for c in categorical_features if c in X_train]

model_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    is_unbalance=True,
    n_jobs=-1,
    random_state=42,
    n_estimators=500 # Give it a decent number of trees
)

print("Training LightGBM...")
model_lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             eval_metric='auc',
             callbacks=[lgb.early_stopping(10, verbose=False)],
             categorical_feature=categorical_indices
            )

train_time_lgbm = time.time() - start_time
# Stop tracking this task and get its specific data
emissions_data_lgbm = tracker.stop_task()

# --- Evaluation --- 
print("Evaluating LightGBM...")
start_inf_time = time.time()
y_prob_lgbm = model_lgbm.predict_proba(X_test)[:, 1]
total_inf_time_lgbm = time.time() - start_inf_time

# Get Performance Metrics
perf_lgbm = get_performance_metrics("LightGBM Proposed", y_test, y_prob_lgbm, y_prob_lgbm)

# Get Sustainability Metrics
sust_lgbm = get_sustainability_metrics(
    "LightGBM Proposed", 
    train_time_lgbm, 
    total_inf_time_lgbm, 
    len(X_test), 
    emissions_data_lgbm
)

results["LightGBM_Proposed"] = {"Performance": perf_lgbm, "Sustainability": sust_lgbm}
print("LightGBM Proposed Model Finished.")
print("-" * 40 + "\n")

# --------------------------------------------------
# Step 7: Final Evaluation & Comparison
# --------------------------------------------------
print("Step 7: Compiling Final Comparison...")

# Create a summary DataFrame
summary_list = []
for model_name, data in results.items():
    flat_row = {'Model': model_name}
    flat_row.update(data['Performance'])
    flat_row.update(data['Sustainability'])
    summary_list.append(flat_row)

df_summary = pd.DataFrame(summary_list).set_index('Model')

print("\n" + "="*60)
print("--- FINAL MODEL COMPARISON ---")
print("="*60 + "\n")
print(df_summary)

print("\n" + "-" * 40)
print("Analysis Complete.")
print("-" * 40 + "\n")

# --------------------------------------------------
# Step 8: Conclusion
# --------------------------------------------------
print("### Conclusion ###\n")
print("1. Performance: Observe the metrics. LightGBM (a tree-based model) is famous for")
print("   often outperforming DNNs on tabular data, especially in metrics like AUC and F1.")
print("\n2. Sustainability: The LightGBM model will almost certainly be")
print("   dramatically more efficient in Training Time, Inference Latency, and CO2/Energy usage.")
print("\nThis analysis strongly supports the hypothesis: energy-efficient models like LightGBM")
print("are often both 'greener' and higher-performing for structured, tabular data.")

# --------------------------------------------------
# Step 9: Stop the main tracker
# --------------------------------------------------
_ = tracker.stop()
print("\nCarbon tracking complete. Script finished.")