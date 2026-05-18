
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("FEAST FEATURE STORE SETUP")
print("=" * 60)

# Step 1: Load prepared data
print("\n[1] Loading prepared data...")
df = pd.read_csv("prepared_data/airline_cleaned.csv")
print(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Step 2: Add Feast-required columns
print("\n[2] Adding entity key and timestamp...")
df["passenger_id"] = range(1, len(df) + 1)
df["event_timestamp"] = pd.Timestamp(datetime.now())

# Step 3: Rename columns (Feast doesn't like spaces/slashes in column names)
print("\n[3] Renaming columns for Feast compatibility...")
rename_map = {
    "Seat comfort": "seat_comfort",
    "Departure/Arrival time convenient": "departure_arrival_time_convenient",
    "Food and drink": "food_and_drink",
    "Gate location": "gate_location",
    "Inflight wifi service": "inflight_wifi_service",
    "Inflight entertainment": "inflight_entertainment",
    "Online support": "online_support",
    "Ease of Online booking": "ease_of_online_booking",
    "On-board service": "onboard_service",
    "Leg room service": "leg_room_service",
    "Baggage handling": "baggage_handling",
    "Checkin service": "checkin_service",
    "Cleanliness": "cleanliness",
    "Online boarding": "online_boarding",
    "Age": "age",
    "Flight Distance": "flight_distance",
    "Departure Delay in Minutes": "departure_delay",
    "Arrival Delay in Minutes": "arrival_delay",
    "Total_Service_Score": "total_service_score",
    "Total_Delay": "total_delay",
    "Is_Long_Flight": "is_long_flight",
    "Customer Type_Loyal Customer": "customer_type_loyal",
    "Customer Type_disloyal Customer": "customer_type_disloyal",
    "Type of Travel_Business travel": "travel_type_business",
    "Type of Travel_Personal Travel": "travel_type_personal",
    "Class_Business": "class_business",
    "Class_Eco": "class_eco",
    "Class_Eco Plus": "class_eco_plus",
}
df.rename(columns=rename_map, inplace=True)

# Convert boolean one-hot columns to int
bool_cols = [
    "customer_type_loyal", "customer_type_disloyal",
    "travel_type_business", "travel_type_personal",
    "class_business", "class_eco", "class_eco_plus",
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Step 4: Save as Parquet
print("\n[4] Saving Parquet file...")
parquet_path = "data/airline_features.parquet"
df.to_parquet(parquet_path, index=False)
print(f"    Saved: {parquet_path} ({df.shape[0]:,} rows x {df.shape[1]} columns)")

# Step 5: Print column summary
print("\n[5] Parquet columns:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:2d}. {col} ({df[col].dtype})")

print("\n" + "=" * 60)
print("PARQUET READY")
