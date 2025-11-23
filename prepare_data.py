"""
Quick script to prepare weather_energy.csv for streaming app
"""

import pandas as pd
import numpy as np
import os
from glob import glob

print("Preparing weather_energy.csv...")

# Create temp_files directory
os.makedirs('temp_files', exist_ok=True)

# Load energy data from all blocks
print("Loading energy data blocks...")
block_files = glob('dataset/daily_dataset/daily_dataset/*.csv')
energy_list = []

for file in block_files:
    df = pd.read_csv(file)
    energy_list.append(df)

energy = pd.concat(energy_list, ignore_index=True)
print(f"Loaded {len(energy)} energy records")

# Group by day
energy_grouped = energy.groupby('day').agg({
    'energy_sum': 'sum',
    'LCLid': 'count'
}).reset_index()

energy_grouped['avg_energy'] = energy_grouped['energy_sum'] / energy_grouped['LCLid']
energy_grouped['day'] = pd.to_datetime(energy_grouped['day'])

# Load weather data
print("Loading weather data...")
weather = pd.read_csv('dataset/weather_daily_darksky.csv')
weather['time'] = pd.to_datetime(weather['time'])
weather.rename(columns={'time': 'day'}, inplace=True)

# Merge
print("Merging datasets...")
weather_energy = pd.merge(energy_grouped, weather, on='day', how='inner')

# Add date features
weather_energy['year'] = weather_energy['day'].dt.year
weather_energy['month'] = weather_energy['day'].dt.month

# Load holidays
holidays = pd.read_csv('dataset/uk_bank_holidays.csv')
holidays['Bank holidays'] = pd.to_datetime(holidays['Bank holidays'])
holidays['holiday_ind'] = 1

weather_energy = pd.merge(
    weather_energy,
    holidays,
    left_on='day',
    right_on='Bank holidays',
    how='left'
)
weather_energy['holiday_ind'] = weather_energy['holiday_ind'].fillna(0).astype(int)

# K-means clustering for weather
from sklearn.cluster import KMeans

weather_features = weather_energy[['temperatureMax', 'humidity', 'windSpeed']].copy()
weather_features = weather_features.fillna(weather_features.mean())

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
weather_energy['weather_cluster'] = kmeans.fit_predict(weather_features)

# Select important columns
columns_to_keep = [
    'day', 'avg_energy', 'temperatureMax', 'humidity', 'windSpeed',
    'weather_cluster', 'holiday_ind', 'year', 'month'
]

weather_energy = weather_energy[columns_to_keep].copy()
weather_energy = weather_energy.sort_values('day').reset_index(drop=True)

# Save
output_path = 'temp_files/weather_energy.csv'
weather_energy.to_csv(output_path, index=False)

print(f"\nSuccess! Saved {len(weather_energy)} rows to {output_path}")
print(f"Date range: {weather_energy['day'].min()} to {weather_energy['day'].max()}")
print(f"\nColumns: {list(weather_energy.columns)}")
print(f"\nFirst few rows:")
print(weather_energy.head())
