"""
Data Preparation Script for Airline Customer Satisfaction
Handles: missing values, encoding, outlier capping, feature engineering, scaling
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'airline.csv'
OUTPUT_DIR = 'prepared_data'
LOG_FILE = os.path.join(OUTPUT_DIR, 'data_prep_log.txt')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize logger
log_messages = []

def log_message(msg):
    """Append message to log and print"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    log_messages.append(full_msg)
    print(full_msg)

def main():
    log_message("="*80)
    log_message("AIRLINE DATA PREPARATION PIPELINE")
    log_message("="*80)
    
    # Step 1: Load data
    log_message(f"\n[STEP 1] Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    log_message(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    log_message(f"  Initial memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Step 2: Log initial nulls
    log_message(f"\n[STEP 2] Checking for missing values...")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        log_message(f"  Found missing values:")
        for col, count in null_counts[null_counts > 0].items():
            log_message(f"    - {col}: {count} nulls ({count/len(df)*100:.2f}%)")
    else:
        log_message(f"  ✓ No missing values found")
    
    # Step 3: Handle missing values
    log_message(f"\n[STEP 3] Handling missing values...")
    if df['Arrival Delay in Minutes'].isnull().sum() > 0:
        median_arrival_delay = df['Arrival Delay in Minutes'].median()
        df['Arrival Delay in Minutes'].fillna(median_arrival_delay, inplace=True)
        log_message(f"  ✓ Filled Arrival Delay with median: {median_arrival_delay:.2f}")
    
    # Check for missing values in service-rating columns
    service_cols = [
        'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink',
        'Gate location', 'Inflight wifi service', 'Inflight entertainment',
        'Online support', 'Ease of Online booking', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Cleanliness', 'Online boarding'
    ]
    service_null_counts = df[service_cols].isnull().sum()
    if service_null_counts.sum() > 0:
        log_message(f"  Found missing values in service columns:")
        for col, count in service_null_counts[service_null_counts > 0].items():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log_message(f"    - {col}: {count} nulls → filled with median {median_val:.2f}")
    else:
        log_message(f"  ✓ No missing values found in service-rating columns")
    
    # Step 4: Convert target to binary
    log_message(f"\n[STEP 4] Converting target variable...")
    log_message(f"  Before: {df['satisfaction'].unique()}")
    df['satisfaction'] = (df['satisfaction'] == 'satisfied').astype(int)
    log_message(f"  After: satisfied=1, dissatisfied=0")
    log_message(f"  Class distribution: {df['satisfaction'].value_counts().to_dict()}")
    
    # Step 5: Store original before encoding for reference
    df_original = df.copy()
    
    # Step 6: One-hot encode categorical variables
    log_message(f"\n[STEP 5] Encoding categorical variables...")
    cat_cols = ['Customer Type', 'Type of Travel', 'Class']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    log_message(f"  ✓ One-hot encoded: {cat_cols}")
    log_message(f"  ✓ Converted {len(bool_cols)} boolean columns to int (0/1)")
    log_message(f"  Resulting shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")
    
    # Step 7: Cap outlier delays at 95th percentile
    log_message(f"\n[STEP 6] Handling delay outliers...")
    for col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        p95 = df_encoded[col].quantile(0.95)
        outlier_count = (df_encoded[col] > p95).sum()
        df_encoded[col] = df_encoded[col].clip(upper=p95)
        log_message(f"  ✓ {col}: capped {outlier_count} values at 95th percentile ({p95:.0f} min)")
    
    # Step 8: Feature engineering
    log_message(f"\n[STEP 7] Feature engineering...")
    
    # Total Service Score (average of 14 service ratings)
    service_cols = [
        'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink',
        'Gate location', 'Inflight wifi service', 'Inflight entertainment',
        'Online support', 'Ease of Online booking', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Cleanliness', 'Online boarding'
    ]
    df_encoded['Total_Service_Score'] = df_encoded[service_cols].mean(axis=1)


    ONLINE_COLS = ['Online boarding', 'Online support', 'Ease of Online booking']
    df['online_experience_score'] = df[ONLINE_COLS].mean(axis=1)

    INFLIGHT_COLS = ['Seat comfort', 'Food and drink', 'Inflight entertainment',
                     'Leg room service', 'On-board service', 'Cleanliness', 'Inflight wifi service']
    df['inflight_experience_score'] = df[INFLIGHT_COLS].mean(axis=1)

    GROUND_COLS = ['Gate location', 'Departure/Arrival time convenient',
                   'Checkin service', 'Baggage handling']
    df['ground_experience_score'] = df[GROUND_COLS].mean(axis=1)
   
    df_encoded['Total_Delay'] = df_encoded['Departure Delay in Minutes'] + df_encoded['Arrival Delay in Minutes']
    log_message(f"  ✓ Created new features Total_Service_Score, online_experience_score, inflight_experience_score, ground_experience_score, Total_Delay")
    # Is Long Flight (data-driven threshold selection)
    log_message("\n[STEP 7.a] Selecting data-driven threshold for Is_Long_Flight...")
    # Use df_original (which contains numeric `satisfaction`) to evaluate candidate cutoffs
    candidate_q = [0.75, 0.80, 0.85, 0.90, 0.95]
    best_q = None
    best_diff = -1.0
    best_t = None
    for q in candidate_q:
        t = df_original['Flight Distance'].quantile(q)
        high_mean = df_original.loc[df_original['Flight Distance'] > t, 'satisfaction'].mean()
        low_mean = df_original.loc[df_original['Flight Distance'] <= t, 'satisfaction'].mean()
        diff = abs(high_mean - low_mean)
        log_message(f"    candidate q={q:.2f} t={int(t)} diff={diff:.4f}")
        if diff > best_diff:
            best_diff = diff
            best_q = q
            best_t = t

    # Use the best threshold found
    flight_distance_threshold = int(best_t) if best_t is not None else 2500
    df_encoded['Is_Long_Flight'] = (df_encoded['Flight Distance'] > flight_distance_threshold).astype(int)
    long_count = int(df_encoded['Is_Long_Flight'].sum())
    log_message(f"  ✓ Created Is_Long_Flight using data-driven threshold: >{flight_distance_threshold} km (quantile={best_q}, diff={best_diff:.4f})")
    log_message(f"    Long flights: {long_count} ({long_count/len(df_encoded)*100:.2f}%)")
    
    # Step 9: Scale numeric features
    log_message(f"\n[STEP 8] Scaling numeric features...")
    numeric_cols_to_scale = [
        'Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
        'Total_Service_Score', 'Total_Delay'
    ]

    scaler = MinMaxScaler()
    df_encoded[numeric_cols_to_scale] = scaler.fit_transform(df_encoded[numeric_cols_to_scale])
    log_message(f"  ✓ Applied MinMaxScaler to {len(numeric_cols_to_scale)} columns")
    log_message(f"    Columns scaled: {numeric_cols_to_scale}")

    # Step 10: Save outputs
    log_message(f"\n[STEP 9] Saving outputs...")

    cleaned_file = os.path.join(OUTPUT_DIR, 'airline_cleaned.csv')
    df_encoded.to_csv(cleaned_file, index=False)

    log_message(f" Saved: {cleaned_file} ({df_encoded.shape[0]:,} rows)")
    
    # Step 12: Final summary
    log_message(f"\n[SUMMARY] Data Preparation Complete")
    log_message(f"  Total rows processed: {df.shape[0]:,}")
    log_message(f"  Total columns after preparation: {df_encoded.shape[1]}")
    log_message(f"  Features engineered: 3 (Total_Service_Score, Total_Delay, Is_Long_Flight)")
    log_message(f"  Features scaled: {len(numeric_cols_to_scale)}")
    log_message(f"  Categorical features one-hot encoded: {len(cat_cols)}")
    log_message(f"  Output directory: {OUTPUT_DIR}/")
    log_message("="*80)
    
    # Save log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_messages))
    log_message(f"\n  ✓ Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
