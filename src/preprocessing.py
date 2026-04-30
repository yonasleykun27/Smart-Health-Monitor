import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config

def full_preprocessing_and_save(file_path=config.DATA_PATH, output_dir=config.PROCESSED_DATA_DIR):
    """
    Performs end-to-end preprocessing: Loading, Feature Engineering, 
    Outlier Handling, Scaling, and Saving.
    """
    
    # --- STEP 1: LOAD DATA ---
    # Load the raw dataset from the path defined in config.py
    df = pd.read_csv(file_path)
    
    # --- STEP 2: FEATURE ENGINEERING (New Feature) ---
    # We create a combined score of Age and BMI. 
    # High age + High BMI usually indicates higher health risk.
    if 'age' in df.columns and 'bmi' in df.columns:
        df['age_bmi_score'] = df['age'] * df['bmi']
    
    # --- STEP 3: DEFINE TARGET (Label) ---
    # We take the maximum value from health indicators (like heart_disease, stroke)
    # to create a single 'target' column for Supervised Learning.
    df['target'] = df[config.TARGET_CONSTITUENTS].max(axis=1)
    
    # --- STEP 4: CATEGORICAL ENCODING ---
    # Convert 'gender' (Male/Female) into numbers (0/1) so the LSTM can process it.
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender']) 
    
    # --- STEP 5: HANDLE MISSING VALUES ---
    # Fill any empty cells with the Median. Median is better than Mean 
    # because it is not affected by extreme outliers.
    df = df.fillna(df.median(numeric_only=True))
    
    # --- STEP 6: OUTLIER HANDLING (CLIPPING) ---
    # Caps extreme values at 1st and 99th percentiles.
    # This prevents the model from being confused by "impossible" data points.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(
        lower=df[numeric_cols].quantile(0.01), 
        upper=df[numeric_cols].quantile(0.99), 
        axis=1
    )
    
    # --- STEP 7: SKEWNESS HANDLING (LOG TRANSFORM) ---
    # We apply Log Transformation to 'glucose' to normalize its distribution.
    # log1p is used because it handles 0 values safely (log(1+x)).
    if 'glucose' in df.columns:
        df['glucose'] = np.log1p(df['glucose'])
    
    # --- STEP 8: FEATURE SELECTION (X and Y) ---
    # X = Input features (clues), y = Target (the answer).
    X = df.drop(columns=config.DROP_COLUMNS + ['target'])
    y = df['target']
    
    # --- STEP 9: TRAIN/TEST SPLIT ---
    # Split data: 80% for training the AI, 20% for testing its accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # --- STEP 10: NORMALIZATION (SCALING) ---
    # Scaler transforms data to have a Mean of 0 and Std Dev of 1.
    # CRITICAL: We only 'fit' on Train data to avoid data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- STEP 11: SAVING DATA ---
    # Create the processed directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save as .npy files for high-speed loading during Model Training.
    np.save(f'{output_dir}X_train.npy', X_train_scaled)
    np.save(f'{output_dir}X_test.npy', X_test_scaled)
    np.save(f'{output_dir}y_train.npy', y_train.values)
    np.save(f'{output_dir}y_test.npy', y_test.values)

    print(f"✅ Advanced processing complete! {X_train.shape[1]} features ready.")
    print(f"📁 Processed files saved in: {output_dir}")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values