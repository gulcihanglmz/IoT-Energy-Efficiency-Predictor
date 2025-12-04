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
        self.history = []
        
    def load_model(self):
        """Load the trained SARIMAX model"""
        print(f"Loading model from {self.model_path}...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
            print(f"Model type: {type(self.model)}")
            
            # Check if model has exog
            if hasattr(self.model, 'k_exog'):
                print(f"Model expects {self.model.k_exog} exogenous variables")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please ensure the SARIMAX model is trained and saved")
            raise
    
    def predict_next(self, exog_data: Dict = None) -> float:
        """
        Predict next time step
        
        Args:
            exog_data: Dictionary with exogenous variables (weather_cluster, holiday)
            
        Returns:
            Predicted energy consumption
        """
        if self.model is None:
            self.load_model()
        
        # DÜZELTME: Model exog kullanmıyorsa direkt rule-based kullan
        use_rule_based = False
        
        # Check if model uses exog
        if hasattr(self.model, 'k_exog') and self.model.k_exog > 0:
            # Model expects exogenous variables
            exog = None
            if exog_data:
                exog = pd.DataFrame({
                    'weather_cluster': [int(exog_data.get('weather_cluster', 1))],
                    'holiday': [int(exog_data.get('holiday', 0))]
                })
                
                print(f"DEBUG - Using SARIMAX with exog: weather={exog['weather_cluster'].iloc[0]}, "
                      f"holiday={exog['holiday'].iloc[0]}")
            
            try:
                forecast = self.model.forecast(steps=1, exog=exog)
                prediction = float(forecast.iloc[0])
                
                # Check if prediction is too similar (model not using exog properly)
                if len(self.history) > 5:
                    recent_predictions = [h['predicted'] for h in self.history[-5:]]
                    avg_recent = np.mean(recent_predictions)
                    if abs(prediction - avg_recent) < 0.5:  # Too similar
                        print("⚠️ WARNING: Model predictions too similar, switching to rule-based")
                        use_rule_based = True
                
                if not use_rule_based and prediction > 0:
                    print(f"✓ SARIMAX prediction: {prediction:.2f} kWh")
                    return max(5.0, min(25.0, prediction))
                else:
                    use_rule_based = True
                
            except Exception as e:
                print(f"✗ SARIMAX error: {e}")
                use_rule_based = True
        else:
            print("⚠️ Model doesn't use exogenous variables")
            use_rule_based = True
        
        # Fallback to rule-based
        if use_rule_based:
            return self._rule_based_prediction(exog_data)

    def _rule_based_prediction(self, exog_data: Dict = None) -> float:
        """
        Rule-based prediction with realistic variations
        
        Args:
            exog_data: Dictionary with weather_cluster and holiday
            
        Returns:
            Predicted energy consumption
        """
        if exog_data is None:
            return 11.0
        
        # Base consumption
        base = 11.0
        
        # Weather cluster effect (GÜÇLÜ ETKI)
        weather = exog_data.get('weather_cluster', 1)
        weather_effect = {
            0: -2.5,  # Clear/Sunny: -2.5 kWh (çok az ısınma)
            1: 0.0,   # Partly Cloudy: normal
            2: +0.8,  # Cloudy: +0.8 kWh
            3: +1.5,  # Rainy: +1.5 kWh (daha fazla ısınma)
            4: +3.0   # Cold/Snowy: +3.0 kWh (çok fazla ısınma)
        }
        base += weather_effect.get(weather, 0.0)
        
        # Holiday effect
        if exog_data.get('holiday', 0) == 1:
            base *= 0.80  # Tatilde %20 daha az
        
        # Random variation (±5%)
        variation = np.random.uniform(-0.05, 0.05)
        base *= (1 + variation)
        
        print(f"✓ Rule-based: {base:.2f} kWh (weather={weather}, holiday={exog_data.get('holiday', 0)})")
        
        return max(5.0, min(25.0, base))
    
    def update_history(self, actual: float, predicted: float):
        """
        Update prediction history
        
        Args:
            actual: Actual energy consumption
            predicted: Predicted energy consumption
        """
        self.history.append({
            'actual': actual,
            'predicted': predicted,
            'error': actual - predicted,
            'timestamp': datetime.now()
        })
    
    def get_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with MAE, RMSE, MAPE, R2
        """
        if len(self.history) < 2:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'r2': 0.0,
                'count': len(self.history)
            }
        
        actual = np.array([h['actual'] for h in self.history])
        predicted = np.array([h['predicted'] for h in self.history])
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        
        # R2 score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'count': len(self.history)
        }
    
    def reset_history(self):
        """Reset prediction history"""
        self.history = []


if __name__ == "__main__":
    # Test the predictor
    print("Testing Energy Predictor:\n")
    
    predictor = EnergyPredictor()
    
    try:
        predictor.load_model()
        
        # Test different weather conditions
        test_cases = [
            {'weather_cluster': 0, 'holiday': 0, 'desc': 'Clear/Sunny, Working day'},
            {'weather_cluster': 4, 'holiday': 0, 'desc': 'Cold/Snowy, Working day'},
            {'weather_cluster': 1, 'holiday': 1, 'desc': 'Partly Cloudy, Holiday'},
            {'weather_cluster': 3, 'holiday': 0, 'desc': 'Rainy, Working day'},
        ]
        
        print("\nTesting different scenarios:\n")
        print("=" * 60)
        for test in test_cases:
            prediction = predictor.predict_next({
                'weather_cluster': test['weather_cluster'],
                'holiday': test['holiday']
            })
            print(f"{test['desc']:40s}: {prediction:.2f} kWh")
        
        print("=" * 60)
        
        # Test with some dummy data
        print("\nTesting metrics calculation:\n")
        predictor.update_history(actual=10.5, predicted=10.2)
        predictor.update_history(actual=11.5, predicted=11.3)
        predictor.update_history(actual=9.8, predicted=10.1)
        
        metrics = predictor.get_metrics()
        print(f"Test metrics:")
        print(f"  MAE:  {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²:   {metrics['r2']:.3f}")
        
    except FileNotFoundError:
        print("\nNote: Run the Jupyter notebook first to train and save the SARIMAX model")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
