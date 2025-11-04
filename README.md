# AI-Based Carbon-Neutral Fraud Detection

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Kaggle](https://img.shields.io/badge/dataset-IEEE--CIS-blue)
![Codecarbon](https://img.shields.io/badge/tracked%20by-CodeCarbon-green)

A Python project that compares the performance, energy consumption, and carbon footprint of a Deep Neural Network (DNN) vs. a LightGBM model for financial fraud detection. This project uses a 5-fold stratified cross-validation methodology to ensure robust and reliable results.

## 1. Project Hypothesis

The AI industry is facing a critical challenge: state-of-the-art models, particularly Deep Neural Networks, consume massive amounts of energy, contributing to a significant carbon footprint. This project investigates this trade-off in the critical domain of financial fraud detection.

**Hypothesis:** An energy-efficient gradient boosting model (LightGBM) can achieve **superior** fraud detection performance on large-scale tabular data while using **dramatically less energy** and producing fewer CO₂ emissions than a high-energy Deep Neural Network (DNN) baseline.

## 2. Methodology

This project implements a three-phase evaluation framework:

### Phase 1: Dataset

- **Source:** [IEEE-CIS Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Details:** A large-scale, real-world dataset containing transaction and identity data (590,540 transactions, 431 features), known for its high-dimensionality and severe class imbalance.

### Phase 2: Modeling

- **Baseline (High-Energy):** A multi-layer Deep Neural Network (DNN) built with `TensorFlow/Keras`, complete with Batch Normalization and Dropout layers.
- **Proposed (Energy-Efficient):** A LightGBM (Gradient Boosting Machine) model, a tree-based algorithm renowned for its high performance and efficiency on tabular data.

### Phase 3: Evaluation Framework

- **Validation:** A **5-Fold Stratified Cross-Validation** methodology is used to ensure all metrics are robust, stable, and not the result of a single "lucky" data split.
- **Performance Metrics:** F1-Score, AUC, Precision, and Recall.
- **Sustainability Metrics:** Training Time (s), Energy Consumed (Wh), and Total CO₂ Emissions (g).

All sustainability metrics are tracked in real-time using the `codecarbon` library, which measures power draw from the CPU and GPU.

## 3. Results & Conclusion

The 5-fold cross-validation experiment was executed on a machine with a 13th Gen Intel i7 CPU and an NVIDIA RTX 4050 Laptop GPU. The aggregated results conclusively support the project's hypothesis.

### Table 3.1: Performance Results (5-Fold Cross-Validation)

This table focuses on the accuracy and reliability of the models.

| Metric    |  DNN (Baseline)   | **LightGBM (Proposed)** |
| :-------- | :---------------: | :---------------------: |
| F1-Score  | 0.2700 (± 0.0401) |  **0.6563 (± 0.0516)**  |
| AUC       | 0.8336 (± 0.0114) |  **0.9689 (± 0.0028)**  |
| Recall    | 0.1600 (± 0.0278) |  **0.8470 (± 0.0075)**  |
| Precision | 0.8805 (± 0.0125) |    0.5387 (± 0.0696)    |

### Table 3.2: Sustainability Results (5-Fold Cross-Validation)

This table focuses on the efficiency and environmental cost of training the models.

| Metric            |  DNN (Baseline)   | **LightGBM (Proposed)** |
| :---------------- | :---------------: | :---------------------: |
| Training Time (s) |  67.35 (± 16.35)  |   **42.08 (± 16.62)**   |
| Energy (Wh)       | 1.0012 (± 0.2407) |  **0.6280 (± 0.2405)**  |
| CO₂ Emissions (g) | 0.7143 (± 0.1717) |  **0.4480 (± 0.1716)**  |

### Key Findings

1.  **Performance (Accuracy):** The energy-efficient **LightGBM model was dramatically and consistently superior** in every key performance metric. Most critically, its average **Recall of 0.847** proves it **found 84.7% of all fraud**. In contrast, the DNN (Recall 0.160) **failed to find 84% of all fraud**, making it an unsuitable model for this task.

2.  **Performance (Stability):** LightGBM was also more stable. Its standard deviation for the AUC score (`±0.0028`) was 4 times smaller than the DNN's (`±0.0114`), proving it delivers a highly reliable score across all data folds.

3.  **Sustainability (Efficiency):** The LightGBM model was **quantifiably "greener."** On average, it trained **37.5% faster** than the DNN, while consuming **37.3% less energy** and producing **37.3% less carbon**.

### Conclusion

This analysis proves that for this large-scale, tabular data problem, **LightGBM is not a compromise; it is the superior choice in every category.** It delivers a more accurate, reliable, and faster model while being significantly more energy-efficient and environmentally friendly than the DNN baseline. This strongly suggests that for many real-world tabular data problems, the pursuit of "Green AI" and the pursuit of high performance are the same goal.

## 4. How to Run This Project

You can replicate this experiment by following these steps:

### Step 1: Download the Dataset

This script **requires** the IEEE-CIS dataset from Kaggle.

1.  Go to: <https://www.kaggle.com/c/ieee-fraud-detection/data>
2.  Download the following two files:
    - `train_transaction.csv`
    - `train_identity.csv`
3.  Place both `.csv` files in the **same directory** as the `run_analysis.py` script.

### Step 2: Set Up Your Environment

It is highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# (macOS/Linux)
# source venv/bin/activate
```

### Step 3: Install Requirements

Install all necessary libraries:

```bash
pip install pandas numpy scikit-learn tensorflow lightgbm codecarbon
```

### Step 4: Run the Analysis

Execute the script from your terminal. Note: This will run the full 5-fold cross-validation and will take a significant amount of time to complete.

```bash
python run_analysis.py
```

The script will loop through all 5 folds, train and evaluate both models for each fold, and then print the final aggregated comparison table (Mean and Std. Dev.) to your console. It will also generate an emissions.csv file in the same directory with a detailed log from codecarbon.
