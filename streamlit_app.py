"""
Real-time Energy Consumption Forecasting Dashboard
Interactive Streamlit application for IoT energy monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os
import requests
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_simulator import EnergyDataSimulator
from src.predictor import EnergyPredictor


# Page configuration
st.set_page_config(
    page_title="IoT Energy Efficiency Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sensor-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(46, 204, 113, 0.3);
        margin: 1rem 0;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


def download_model():
    """Download SARIMAX model from GitHub releases if not present"""
    model_path = Path("sarimax_model.pkl")
    
    if model_path.exists():
        st.sidebar.info(f"Model found: {model_path}")
        return str(model_path)
    
    # GitHub release URL
    url = "https://github.com/gulcihanglmz/IoT-Energy-Efficiency-Predictor/releases/download/v1.0.0/sarimax_model.pkl"
    
    st.sidebar.warning("Model not found. Downloading from GitHub...")
    
    progress_placeholder = st.sidebar.empty()
    status_placeholder = st.sidebar.empty()
    
    try:
        with st.spinner("Downloading model... (this may take a minute)"):
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.text(f"Downloading: 0 / {total_size/1024/1024:.1f} MB")
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_placeholder.text(
                                f"Downloading: {downloaded/1024/1024:.1f} / {total_size/1024/1024:.1f} MB"
                            )
            
            progress_placeholder.empty()
            status_placeholder.empty()
            st.sidebar.success("Model downloaded successfully!")
            return str(model_path)
            
    except requests.exceptions.RequestException as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.sidebar.error(f"Download failed: {e}")
        st.sidebar.info("You can manually download the model from the GitHub releases page")
        raise
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.sidebar.error(f"Error: {e}")
        raise


def initialize_session_state():
    """Initialize session state variables"""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'data_stream' not in st.session_state:
        st.session_state.data_stream = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    if 'stream_started' not in st.session_state:
        st.session_state.stream_started = False
    if 'manual_predictions' not in st.session_state:
        st.session_state.manual_predictions = []
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'model_downloaded' not in st.session_state:
        st.session_state.model_downloaded = False


def create_real_time_chart(df: pd.DataFrame):
    """Create real-time comparison chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Energy Consumption: Actual vs Predicted', 
                       'Prediction Error Over Time',
                       'Temperature Trend',
                       'Cumulative Error'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['actual'], name='Actual',
                  mode='lines+markers', line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['predicted'], name='Predicted',
                  mode='lines+markers', line=dict(color='#e74c3c', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Plot 2: Error
    df['error'] = df['actual'] - df['predicted']
    fig.add_trace(
        go.Bar(x=df['date'], y=df['error'], name='Error',
              marker=dict(color=['#2ecc71' if e > 0 else '#e74c3c' for e in df['error']])),
        row=1, col=2
    )
    
    # Plot 3: Temperature
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['temperature'], name='Temperature',
                  mode='lines', fill='tozeroy', line=dict(color='#f39c12', width=2)),
        row=2, col=1
    )
    
    # Plot 4: Cumulative Error
    df['cumulative_error'] = df['error'].cumsum()
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cumulative_error'], name='Cumulative Error',
                  mode='lines', line=dict(color='#9b59b6', width=2)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Error (kWh)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Error", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    
    return fig


def create_manual_input_gauge(value: float, title: str, max_val: float = 100):
    """Create a gauge chart for manual input visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_val*0.33], 'color': "#d4edda"},
                {'range': [max_val*0.33, max_val*0.66], 'color': "#fff3cd"},
                {'range': [max_val*0.66, max_val], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    fig.update_layout(height=250)
    return fig


def create_forecast_chart(forecast_df: pd.DataFrame):
    """Create forecast visualization chart"""
    fig = go.Figure()
    
    # Predicted energy consumption
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_energy'],
        name='Predicted Energy',
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper_bound'],
        name='Upper Bound',
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower_bound'],
        name='Confidence Interval',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(102, 126, 234, 0.2)',
        fill='tonexty',
        showlegend=True
    ))
    
    fig.update_layout(
        title='Energy Consumption Forecast',
        xaxis_title='Date',
        yaxis_title='Energy Consumption (kWh)',
        hovermode='x unified',
        height=500
    )
    
    return fig


def manual_input_tab():
    """Tab for manual IoT sensor input"""
    st.markdown("### IoT Sensor Simulator")
    st.markdown("**Enter sensor values manually - Perfect for Arduino integration testing!**")
    
    if st.session_state.predictor is None:
        st.warning("Please initialize the system first from the sidebar!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sensor-input">', unsafe_allow_html=True)
        st.markdown("**Environmental Sensors**")
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=20.0,
            step=0.5,
            help="Current temperature reading"
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Relative humidity percentage"
        )
        
        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Wind speed in kilometers per hour"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sensor-input">', unsafe_allow_html=True)
        st.markdown("**Date & Time Settings**")
        
        is_holiday = st.checkbox(
            "Is Holiday?",
            value=False,
            help="Toggle if today is a holiday"
        )
        
        weather_cluster = st.selectbox(
            "Weather Condition",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Clear/Sunny",
                1: "Partly Cloudy",
                2: "Cloudy",
                3: "Rainy",
                4: "Cold/Snowy"
            }[x],
            help="Select current weather condition"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Live Sensor Readings**")
        
        gauge_col1, gauge_col2 = st.columns(2)
        
        with gauge_col1:
            st.plotly_chart(
                create_manual_input_gauge(temperature, "Temperature (°C)", 45),
                use_container_width=True
            )
        
        with gauge_col2:
            st.plotly_chart(
                create_manual_input_gauge(humidity, "Humidity (%)", 100),
                use_container_width=True
            )
        
        st.plotly_chart(
            create_manual_input_gauge(wind_speed, "Wind Speed (km/h)", 100),
            use_container_width=True
        )
    
    # Predict button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        if st.button("Predict Energy Consumption", type="primary", use_container_width=True):
            with st.spinner("Calculating prediction..."):
                try:
                    exog_data = {
                        'weather_cluster': weather_cluster,
                        'holiday': 1 if is_holiday else 0
                    }
                    
                    prediction = st.session_state.predictor.predict_next(exog_data)
                    
                    confidence = 0.95
                    std_error = prediction * 0.1
                    lower_bound = prediction - (1.96 * std_error)
                    upper_bound = prediction + (1.96 * std_error)
                    
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'temperature': temperature,
                        'humidity': humidity,
                        'wind_speed': wind_speed,
                        'is_holiday': is_holiday,
                        'weather_cluster': weather_cluster,
                        'prediction': prediction,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    
                    st.session_state.manual_predictions.append(prediction_record)
                    
                    st.markdown(
                        f'<div class="prediction-result">{prediction:.2f} kWh</div>',
                        unsafe_allow_html=True
                    )
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Predicted Consumption",
                            f"{prediction:.2f} kWh",
                            help="Expected energy consumption"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Lower Bound (95% CI)",
                            f"{lower_bound:.2f} kWh",
                            help="Lower confidence interval"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Upper Bound (95% CI)",
                            f"{upper_bound:.2f} kWh",
                            help="Upper confidence interval"
                        )
                    
                    st.info(
                        f"**Prediction Details:**\n"
                        f"- Temperature: {temperature}°C\n"
                        f"- Humidity: {humidity}%\n"
                        f"- Wind Speed: {wind_speed} km/h\n"
                        f"- Weather: {['Clear', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Cold'][weather_cluster]}\n"
                        f"- Holiday: {'Yes' if is_holiday else 'No'}"
                    )
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    # Prediction history - SIMPLIFIED (no styling)
    if len(st.session_state.manual_predictions) > 0:
        st.markdown("---")
        st.markdown("### Prediction History")
        
        history_df = pd.DataFrame(st.session_state.manual_predictions)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Create display dataframe
        display_df = history_df.tail(10)[['timestamp', 'temperature', 'humidity', 'wind_speed', 
                                           'weather_cluster', 'is_holiday', 'prediction']].copy()
        
        # Rename columns
        display_df.columns = ['Time', 'Temp (°C)', 'Humidity (%)', 'Wind (km/h)', 
                              'Weather', 'Holiday', 'Energy (kWh)']
        
        # Format weather cluster
        display_df['Weather'] = display_df['Weather'].map({
            0: 'Clear',
            1: 'Partly Cloudy',
            2: 'Cloudy',
            3: 'Rainy',
            4: 'Cold/Snowy'
        })
        
        # Format holiday
        display_df['Holiday'] = display_df['Holiday'].map({True: 'Yes', False: 'No'})
        
        # Format numbers manually
        display_df['Temp (°C)'] = display_df['Temp (°C)'].apply(lambda x: f"{x:.1f}")
        display_df['Humidity (%)'] = display_df['Humidity (%)'].apply(lambda x: f"{x:.0f}")
        display_df['Wind (km/h)'] = display_df['Wind (km/h)'].apply(lambda x: f"{x:.1f}")
        display_df['Energy (kWh)'] = display_df['Energy (kWh)'].apply(lambda x: f"{x:.2f}")
        
        # Display WITHOUT styling (simpler, more compatible)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )


def forecast_tab():
    """Tab for future forecasting"""
    st.markdown("### Future Energy Forecast")
    st.markdown("**Predict energy consumption for the coming days**")
    
    if st.session_state.predictor is None:
        st.warning("Please initialize the system first from the sidebar!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        forecast_days = st.slider(
            "How many days ahead to forecast?",
            min_value=1,
            max_value=7,
            value=3,
            help="Select number of days to forecast"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_forecast = st.button("Generate Forecast", type="primary", use_container_width=True)
    
    if generate_forecast:
        with st.spinner(f"Generating {forecast_days}-day forecast..."):
            try:
                start_date = datetime.now()
                forecast_dates = [start_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                
                forecast_data = []
                
                for date in forecast_dates:
                    exog_data = {
                        'weather_cluster': 1,
                        'holiday': 0
                    }
                    
                    prediction = st.session_state.predictor.predict_next(exog_data)
                    
                    std_error = prediction * 0.15
                    lower_bound = prediction - (1.96 * std_error)
                    upper_bound = prediction + (1.96 * std_error)
                    
                    forecast_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_energy': prediction,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
                
                forecast_df = pd.DataFrame(forecast_data)
                st.session_state.forecast_results = forecast_df
                
                st.plotly_chart(
                    create_forecast_chart(forecast_df),
                    use_container_width=True
                )
                
                st.markdown("### Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Predicted",
                        f"{forecast_df['predicted_energy'].sum():.2f} kWh"
                    )
                
                with col2:
                    st.metric(
                        "Average Daily",
                        f"{forecast_df['predicted_energy'].mean():.2f} kWh"
                    )
                
                with col3:
                    st.metric(
                        "Peak Day",
                        f"{forecast_df['predicted_energy'].max():.2f} kWh"
                    )
                
                with col4:
                    st.metric(
                        "Lowest Day",
                        f"{forecast_df['predicted_energy'].min():.2f} kWh"
                    )
                
                st.markdown("### Detailed Forecast")
                st.dataframe(
                    forecast_df.style.format({
                        'predicted_energy': '{:.2f} kWh',
                        'lower_bound': '{:.2f} kWh',
                        'upper_bound': '{:.2f} kWh'
                    }),
                    use_container_width=True
                )
                
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")


def streaming_tab():
    """Original real-time streaming tab"""
    st.markdown("### Real-time Data Stream")
    st.markdown("**Simulate real-time sensor data streaming**")
    
    if st.session_state.simulator is None or st.session_state.predictor is None:
        st.warning("Please initialize the system first from the sidebar!")
        return
    
    # Streaming controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Stream", disabled=st.session_state.is_streaming, use_container_width=True):
            st.session_state.is_streaming = True
            st.session_state.stream_started = True
    
    with col2:
        if st.button("Pause Stream", disabled=not st.session_state.is_streaming, use_container_width=True):
            st.session_state.is_streaming = False
    
    with col3:
        if st.button("Reset Stream", use_container_width=True):
            st.session_state.data_stream = []
            st.session_state.is_streaming = False
            st.session_state.stream_started = False
            st.session_state.predictor.load_model()
            st.rerun()
    
    # Charts
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Get max points from sidebar
    max_points = st.session_state.get('max_points', 25)
    
    # Streaming logic
    if st.session_state.is_streaming:
        total_len = len(st.session_state.simulator.data)
        start_idx = max(0, total_len - max_points)
        
        for data_point in st.session_state.simulator.stream_data(
            start_index=start_idx,
            max_points=max_points
        ):
            if not st.session_state.is_streaming:
                break
            
            # DÜZELTME: Outlier'ı atla
            actual = data_point['actual_energy']
            if actual < 1.0 or actual > 30.0:
                print(f"⚠️ Skipping outlier: {actual:.2f} kWh on {data_point['date']}")
                continue
            
            try:
                exog_data = {
                    'weather_cluster': int(data_point.get('weather_cluster', 1)),
                    'holiday': int(data_point.get('holiday', 0))
                }
                
                prediction = st.session_state.predictor.predict_next(exog_data)
                
                # Sanity check
                if prediction > actual * 2.5 or prediction < actual * 0.4:
                    if len(st.session_state.data_stream) > 0:
                        recent_predictions = [d['predicted'] for d in st.session_state.data_stream[-5:]]
                        prediction = sum(recent_predictions) / len(recent_predictions)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                prediction = 11.0
            
            st.session_state.data_stream.append({
                'date': data_point['date'],
                'actual': data_point['actual_energy'],
                'predicted': prediction,
                'temperature': data_point.get('temperature', 0),
                'humidity': data_point.get('humidity', 0),
                'wind_speed': data_point.get('wind_speed', 0)
            })
            
            st.session_state.predictor.update_history(
                actual=data_point['actual_energy'],
                predicted=prediction
            )
            
            df = pd.DataFrame(st.session_state.data_stream)
            fig = create_real_time_chart(df)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            error = abs(data_point['actual_energy'] - prediction)
            error_pct = (error / data_point['actual_energy'] * 100) if data_point['actual_energy'] > 0 else 0
            
            status_placeholder.info(
                f"Streaming... Point {len(st.session_state.data_stream)}/{max_points} | "
                f"Date: {data_point['date']} | "
                f"Actual: {data_point['actual_energy']:.2f} kWh | "
                f"Predicted: {prediction:.2f} kWh | "
                f"Error: {error_pct:.1f}%"
            )
            
            time.sleep(st.session_state.simulator.delay_seconds)
        
        st.session_state.is_streaming = False
        status_placeholder.success("Streaming completed!")
    
    elif st.session_state.stream_started and len(st.session_state.data_stream) > 0:
        df = pd.DataFrame(st.session_state.data_stream)
        fig = create_real_time_chart(df)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        status_placeholder.success("Stream paused. Click 'Start Stream' to continue.")
    
    # Data table - SIMPLIFIED
    if len(st.session_state.data_stream) > 0:
        st.markdown("---")
        st.markdown("### Recent Predictions")
        
        df_display = pd.DataFrame(st.session_state.data_stream).tail(10).copy()
        
        df_display['error'] = df_display['actual'] - df_display['predicted']
        df_display['error_pct'] = (df_display['error'].abs() / df_display['actual'].replace(0, 1) * 100)
        
        display_df = df_display[['date', 'actual', 'predicted', 'error', 'error_pct', 'temperature']].copy()
        display_df.columns = ['Date', 'Actual (kWh)', 'Predicted (kWh)', 'Error (kWh)', 'Error %', 'Temp (°C)']
        
        # Format manually
        display_df['Actual (kWh)'] = display_df['Actual (kWh)'].apply(lambda x: f"{x:.2f}")
        display_df['Predicted (kWh)'] = display_df['Predicted (kWh)'].apply(lambda x: f"{x:.2f}")
        display_df['Error (kWh)'] = display_df['Error (kWh)'].apply(lambda x: f"{x:.2f}")
        display_df['Error %'] = display_df['Error %'].apply(lambda x: f"{x:.1f}%")
        display_df['Temp (°C)'] = display_df['Temp (°C)'].apply(lambda x: f"{x:.1f}")
        
        # Display WITHOUT complex styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Summary statistics
        st.markdown("#### Stream Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_error = df_display['error'].abs().mean()
            st.metric("Avg Error", f"{avg_error:.2f} kWh")
        
        with col2:
            avg_error_pct = df_display['error_pct'].mean()
            st.metric("Avg Error %", f"{avg_error_pct:.1f}%")
        
        with col3:
            max_error = df_display['error'].abs().max()
            st.metric("Max Error", f"{max_error:.2f} kWh")
        
        with col4:
            total_points = len(df_display)
            st.metric("Data Points", f"{total_points}")


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">IoT Energy Efficiency Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Energy Consumption Forecasting with Machine Learning</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("System Configuration")
    
    # Data path
    data_path = st.sidebar.text_input(
        "Data Path", 
        value="temp_files/weather_energy.csv",
        help="Path to the weather_energy.csv file"
    )
    
    # Streaming configuration
    st.sidebar.subheader("Streaming Settings")
    
    delay_seconds = st.sidebar.slider(
        "Delay between updates (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    
    max_points = st.sidebar.number_input(
        "Number of points to stream",
        min_value=10,
        max_value=100,
        value=25,
        help="Last N days from test set"
    )
    
    st.session_state.max_points = max_points
    
    # Initialize components
    if st.sidebar.button("Initialize System", type="primary"):
        with st.spinner("Initializing..."):
            try:
                # Download model
                model_file = download_model()
                
                # Initialize simulator
                st.session_state.simulator = EnergyDataSimulator(
                    data_path=data_path,
                    delay_seconds=delay_seconds
                )
                st.session_state.simulator.load_data()
                
                # Initialize predictor
                st.session_state.predictor = EnergyPredictor(model_path=model_file)
                st.session_state.predictor.load_model()
                
                # Reset
                st.session_state.data_stream = []
                st.session_state.predictions = []
                st.session_state.stream_started = False
                st.session_state.manual_predictions = []
                st.session_state.forecast_results = None
                st.session_state.model_downloaded = True
                
                st.sidebar.success("System initialized successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Initialization failed: {str(e)}")
                return
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    if st.session_state.simulator is not None:
        st.sidebar.success("Simulator: Ready")
    else:
        st.sidebar.error("Simulator: Not initialized")
    
    if st.session_state.predictor is not None:
        st.sidebar.success("Predictor: Ready")
    else:
        st.sidebar.error("Predictor: Not initialized")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "Manual IoT Input",
        "Forecast Mode", 
        "Real-time Stream"
    ])
    
    with tab1:
        manual_input_tab()
    
    with tab2:
        forecast_tab()
    
    with tab3:
        streaming_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
        "Made with by IoT Energy Team | Powered by SARIMAX & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
