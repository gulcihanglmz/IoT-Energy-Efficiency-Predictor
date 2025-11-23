# IoT Energy Efficiency Predictor

A comprehensive machine learning project for forecasting household energy consumption using smart meter data from London households. This project compares multiple forecasting approaches including time series models (SARIMAX, LSTM) and classical machine learning algorithms.

## Project Overview

This project analyzes energy consumption data from 5,567 London households collected between November 2011 and February 2014 as part of the UK Power Networks Low Carbon London project. The goal is to predict daily energy consumption using weather data and temporal patterns.


https://github.com/user-attachments/assets/fd3efcd6-a154-4360-890d-23f89d76fa1d

## Dataset

- **Source**: London Data Store - Low Carbon London Project
- **Period**: November 2011 - February 2014
- **Households**: 5,567
- **Features**: Energy consumption, weather conditions, UK bank holidays
- **Target**: Average energy consumption per household (kWh)

## Methodology

### 1. Data Preparation
- Combined 112 data blocks into a single dataset
- Normalized energy consumption across inconsistent household counts
- Created daily-level aggregated data

### 2. Feature Engineering
- **Weather Clustering**: K-means clustering (k=3) on weather variables
  - Features: Temperature, Humidity, Wind Speed
- **Holiday Indicator**: Binary flag for UK bank holidays
- **Temporal Features**: Year, Month extracted from date
<img width="1990" height="490" alt="output1" src="https://github.com/user-attachments/assets/e07e58ea-ff0a-4912-a926-8eb6fdd948b7" />
<img width="1990" height="492" alt="output" src="https://github.com/user-attachments/assets/bc086bd1-db28-476f-9ac7-036decf24309" />

### 3. Exploratory Data Analysis
Analyzed relationships between weather conditions and energy consumption:
- Temperature: Strong inverse correlation (higher temp = lower consumption)
- Humidity: Positive correlation with energy usage
- Cloud Cover: Similar trend to energy consumption
- UV Index: Inverse relationship with consumption

### 4. Model Development

#### Time Series Models
- **SARIMAX** (Seasonal ARIMA with Exogenous Variables)
  - Order: (7,1,1) 
  - Seasonal Order: (1,1,0,12)
  - Exogenous Variables: weather_cluster, holiday_ind
  
- **LSTM** (Long Short-Term Memory Network)
  - Architecture: 50 LSTM units + Dense layer
  - Input: 7-day lag features
  - Optimizer: Adam, Loss: MAE

#### Classical Machine Learning Models
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble learning (100 estimators)
- **XGBoost**: Gradient boosting (100 estimators)

## Results

### Model Performance Comparison

| Model | MAE | RMSE | MAPE (%) | R² Score | Training Time (s) |
|-------|-----|------|----------|----------|-------------------|
| **SARIMAX** | **1.244** | **1.539** | **5.27** | **0.87** | - |
| **LSTM** | 3.012 | 3.656 | 15.38 | 0.48 | - |
| Linear Regression | 2.38 | 6.61 | 191.42 | -0.16 | 0.015 |
| Random Forest | 2.59 | 6.38 | 182.32 | -0.08 | 0.17 |
| XGBoost | 3.51 | 8.32 | 177.51 | -0.03 | 0.17 |

<img width="1542" height="975" alt="output" src="https://github.com/user-attachments/assets/01aac53c-6876-467e-86a6-ad476ffab885" />

### Key Findings

1. **SARIMAX achieved the best performance** with only 5.27% MAPE and 0.87 R² score
   - Successfully captures temporal patterns, seasonality, and trends
   - Incorporates exogenous weather and holiday variables

2. **LSTM showed moderate performance** with 15.38% MAPE
   - Deep learning approach captures complex temporal patterns
   - Requires more data and tuning for optimal performance

3. **Classical ML models failed** with negative R² scores
   - Unable to capture time series characteristics
   - Treat each day independently, ignoring autocorrelation
   - High MAPE values (177-191%) indicate poor predictive power

### Why Time Series Models Work Better

Energy consumption forecasting is inherently a time series problem with:
- **Temporal Dependencies**: Today's consumption relates to yesterday's
- **Seasonality**: Clear patterns across months and seasons
- **Trend Components**: Long-term consumption trends
- **Autocorrelation**: Past values influence future predictions

Classical ML models ignore these temporal features, making them unsuitable for this domain.

## Installation

### Prerequisites
- Python 3.11+
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gulcihanglmz/IoT-Energy-Efficiency-Predictor.git
cd IoT-Energy-Efficiency-Predictor
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- tensorflow/keras
- xgboost

## Usage

1. **Data Preparation**: Run the initial cells to combine block files and prepare the dataset

2. **Exploratory Analysis**: Execute weather correlation and clustering cells

3. **Model Training**: Train individual models or run all models sequentially
   - SARIMAX: Cells 69-75
   - LSTM: Cells 79-92
   - Linear Regression: Cells 107-116
   - Random Forest: Cells 117-125
   - XGBoost: Cells 126-134

4. **Model Comparison**: Execute the comparison analysis cells (135-140) to generate performance metrics and visualizations

## Project Structure

```
IoT-Energy-Efficiency-Predictor/
├── energy-consumption-forecast.ipynb  # Main analysis notebook
├── dataset/
│   ├── daily_dataset/                # Energy consumption blocks
│   ├── weather_daily_darksky.csv     # Weather data
│   └── uk_bank_holidays.csv          # Holiday information
├── temp_files/                       # Temporary processing files
├── .gitignore
└── README.md
```

## Model Files

Trained models are saved locally (not tracked in git):
- `sarimax_model.pkl`
- `lstm_model.keras`
- `linear_regression_model.pkl`
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `kmeans_model.pkl`
- `scaler.pkl`

## Future Improvements

- Hyperparameter optimization for LSTM
- Additional exogenous variables (temperature forecasts, economic indicators)
- Real-time prediction pipeline
- Model ensemble techniques
- Extended validation period
- Feature importance analysis

## Technical Details

### ACF/PACF Analysis
- Autocorrelation shows gradual decay
- Partial autocorrelation drops after lag 1
- Indicates AR(1) signature

### Stationarity Testing
- Dickey-Fuller test: p > 0.05 (non-stationary)
- After differencing: p < 0.05 (stationary)
- First-order differencing applied

### Train-Test Split
- Training: 798 days
- Testing: 30 days (last month)
- No data leakage in time series split

## Performance Metrics

- **MAE** (Mean Absolute Error): Average magnitude of prediction errors
- **RMSE** (Root Mean Squared Error): Emphasizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error metric
- **R² Score**: Proportion of variance explained by the model

## Contributing

This is an academic project. For suggestions or improvements, please open an issue or submit a pull request.

## License

This project uses publicly available data from the London Data Store. Please refer to their terms of use for data usage guidelines.

## Acknowledgments

- UK Power Networks for the Low Carbon London project
- London Data Store for making the dataset publicly available
- Dark Sky API for weather data

## Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

---

**Note**: Model files (.pkl, .keras) are excluded from the repository due to size constraints. Run the notebook to generate models locally.
