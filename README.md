# bfrb-wearable-detection
Machine learning project for detecting body-focused repetitive behaviors (BFRBs) from wearable sensor data using LightGBM
# Detecting Body-Focused Repetitive Behaviors (BFRBs) from Wearable Sensor Data

This project applies machine learning to identify Body-Focused Repetitive Behaviors (BFRBs) such as hair pulling, skin picking, and face touching using time-series data from wrist-worn wearable devices.

### 🔍 Overview

- ⌚ Raw data includes accelerometer, temperature, and distance (ToF) signals.
- 🧠 Goal: classify 18 types of behaviors from wearable signals.
- 🏢 Data provided by Child Mind Institute and CMI.

### ⚙️ Technical Highlights

- Cleaned and aggregated time-series sensor data per sequence
- Applied advanced feature engineering (range, ratio, sensor fusion)
- Trained classification model using **LightGBM**
- Used 5-Fold Stratified Cross-Validation for evaluation
- Achieved **~57% Macro F1 Score** on validation set

### 📁 Project Structure

- `notebooks/`: step-by-step Jupyter Notebook
- `utils/`: feature engineering & preprocessing scripts
- `outputs/`: final submission file (`submission.parquet`)

### 🚀 Usage

```bash
pip install -r requirements.txt
