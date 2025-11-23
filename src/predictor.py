"""
Real-time Energy Prediction Service
Loads trained SARIMAX model and makes predictions on streaming data
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class EnergyPredictor:
    """Real-time energy consumption predictor using SARIMAX model"""
    
    def __init__(self, model_path: str = "sarimax_model.pkl"):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained SARIMAX model
        """
        self.model_path = model_path
        self.model = None
        self.prediction_history = []
        self.actual_history = []
        
    def load_model(self):
        """Load the trained SARIMAX model"""
        print(f"Loading model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please ensure the SARIMAX model is trained and saved")
            raise
    
    def predict_next(self, exog_data: Dict = None) -> float:
        """
        Predict next time step
        
        Args:
            exog_data: Dictionary with exogenous variables (weather_cluster, holiday_ind)
            
        Returns:
            Predicted energy consumption
        """
        if self.model is None:
            self.load_model()
        
        # Prepare exogenous variables if provided
        exog = None
        if exog_data:
            exog = [[
                exog_data.get('weather_cluster', 0),
                exog_data.get('holiday', 0)
            ]]
        
        # Make prediction
        try:
            forecast = self.model.forecast(steps=1, exog=exog)
            prediction = float(forecast.iloc[0])
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to mean prediction
            prediction = np.mean(self.actual_history) if self.actual_history else 0.0
        
        return prediction
    
    def update_history(self, actual: float, predicted: float):
        """
        Update prediction and actual history
        
        Args:
            actual: Actual energy consumption
            predicted: Predicted energy consumption
        """
        self.actual_history.append(actual)
        self.prediction_history.append(predicted)
    
    def get_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with MAE, RMSE, MAPE, R2
        """
        if len(self.actual_history) < 2:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'r2': 0.0,
                'count': len(self.actual_history)
            }
        
        actual = np.array(self.actual_history)
        predicted = np.array(self.prediction_history)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R2 score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'count': len(self.actual_history)
        }
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history = []
        self.actual_history = []


if __name__ == "__main__":
    # Test the predictor
    print("Testing Energy Predictor:\n")
    
    predictor = EnergyPredictor()
    
    try:
        predictor.load_model()
        
        # Make a test prediction
        test_exog = {'weather_cluster': 1, 'holiday': 0}
        prediction = predictor.predict_next(test_exog)
        print(f"Test prediction: {prediction:.2f} kWh")
        
        # Test with some dummy data
        predictor.update_history(actual=5.2, predicted=5.0)
        predictor.update_history(actual=5.5, predicted=5.3)
        
        metrics = predictor.get_metrics()
        print(f"\nTest metrics: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")
        
    except FileNotFoundError:
        print("\nNote: Run the Jupyter notebook first to train and save the SARIMAX model")
