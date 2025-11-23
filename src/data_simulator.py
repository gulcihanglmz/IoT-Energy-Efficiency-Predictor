"""
Real-time Data Simulator
Simulates IoT sensor data stream by reading from test dataset
"""

import pandas as pd
import time
from datetime import datetime
from typing import Generator, Dict
import json


class EnergyDataSimulator:
    """Simulates real-time energy consumption data stream"""
    
    def __init__(self, data_path: str, delay_seconds: float = 2.0):
        """
        Initialize the data simulator
        
        Args:
            data_path: Path to the CSV file containing energy data
            delay_seconds: Delay between each data point (simulates real-time)
        """
        self.data_path = data_path
        self.delay_seconds = delay_seconds
        self.data = None
        self.current_index = 0
        
    def load_data(self):
        """Load the dataset"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Ensure date column exists
        if 'day' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['day'])
        elif 'date' not in self.data.columns:
            raise ValueError("Dataset must have 'day' or 'date' column")
        
        self.data = self.data.sort_values('date').reset_index(drop=True)
        print(f"Loaded {len(self.data)} data points")
        
    def stream_data(self, start_index: int = 0, max_points: int = None) -> Generator[Dict, None, None]:
        """
        Stream data points one by one with delay
        
        Args:
            start_index: Starting index in the dataset
            max_points: Maximum number of points to stream (None = all)
            
        Yields:
            Dictionary containing data point
        """
        if self.data is None:
            self.load_data()
        
        self.current_index = start_index
        end_index = min(start_index + max_points, len(self.data)) if max_points else len(self.data)
        
        print(f"\nStarting data stream from index {start_index} to {end_index}")
        print(f"Streaming {end_index - start_index} data points with {self.delay_seconds}s delay\n")
        
        for idx in range(start_index, end_index):
            row = self.data.iloc[idx]
            
            # Create data point dictionary
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                'actual_energy': float(row['avg_energy']) if 'avg_energy' in row else None,
                'temperature': float(row['temperatureMax']) if 'temperatureMax' in row else None,
                'humidity': float(row['humidity']) if 'humidity' in row else None,
                'wind_speed': float(row['windSpeed']) if 'windSpeed' in row else None,
                'weather_cluster': int(row['weather_cluster']) if 'weather_cluster' in row else None,
                'holiday': int(row['holiday_ind']) if 'holiday_ind' in row else 0,
                'index': idx
            }
            
            yield data_point
            
            self.current_index = idx + 1
            time.sleep(self.delay_seconds)
    
    def get_historical_data(self, end_index: int) -> pd.DataFrame:
        """
        Get all historical data up to a specific index
        
        Args:
            end_index: End index (exclusive)
            
        Returns:
            DataFrame with historical data
        """
        if self.data is None:
            self.load_data()
        
        return self.data.iloc[:end_index].copy()


if __name__ == "__main__":
    # Test the simulator
    simulator = EnergyDataSimulator(
        data_path="../temp_files/weather_energy.csv",
        delay_seconds=1.0
    )
    
    # Stream last 10 points as a test
    print("Testing data simulator with last 10 points:\n")
    simulator.load_data()
    start_idx = len(simulator.data) - 10
    
    for data_point in simulator.stream_data(start_index=start_idx, max_points=10):
        print(f"[{data_point['timestamp']}] Date: {data_point['date']}, "
              f"Energy: {data_point['actual_energy']:.2f} kWh, "
              f"Temp: {data_point['temperature']:.1f}°C")
