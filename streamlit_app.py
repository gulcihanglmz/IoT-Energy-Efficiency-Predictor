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
from datetime import datetime
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_model():
    """Download SARIMAX model from GitHub releases if not present"""
    model_path = Path("sarimax_model.pkl")
    
    if model_path.exists():
        return str(model_path)
    
    # GitHub release URL
    url = "https://github.com/gulcihanglmz/IoT-Energy-Efficiency-Predictor/releases/download/v1.0.0/sarimax_model.pkl"
    
    with st.spinner("⬇️ Downloading model... (first time only)"):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(downloaded / total_size)
            
            progress_bar.empty()
            st.success("✓ Model downloaded successfully!")
            return str(model_path)
            
        except Exception as e:
            st.error(f"Error downloading model: {e}")
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


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">IoT Energy Efficiency Predictor</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Energy Consumption Forecasting with Machine Learning</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data path
    data_path = st.sidebar.text_input(
        "Data Path", 
        value="temp_files/weather_energy.csv",
        help="Path to the weather_energy.csv file"
    )
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path",
        value="sarimax_model.pkl",
        help="Path to the trained SARIMAX model"
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
    
    # Initialize components
    if st.sidebar.button("Initialize System", type="primary"):
        with st.spinner("Initializing..."):
            try:
                # Download model if needed
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
                
                # Reset history
                st.session_state.data_stream = []
                st.session_state.predictions = []
                st.session_state.stream_started = False
                
                st.sidebar.success("System initialized successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Initialization failed: {str(e)}")
                return
    
    # Start/Stop streaming
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Start Stream", disabled=st.session_state.simulator is None):
        st.session_state.is_streaming = True
        st.session_state.stream_started = True
    
    if col2.button("Stop Stream"):
        st.session_state.is_streaming = False
    
    # Main content
    if st.session_state.simulator is None or st.session_state.predictor is None:
        st.info("Click 'Initialize System' in the sidebar to start")
        st.markdown("""
        ### Instructions:
        1. Ensure `weather_energy.csv` is in the `temp_files/` directory
        2. Ensure `sarimax_model.pkl` is in the root directory
        3. Click **Initialize System** to load the model and data
        4. Click **Start Stream** to begin real-time simulation
        5. Watch the predictions update in real-time!
        """)
        return
    
    # Charts
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Streaming logic
    if st.session_state.is_streaming:
        # Get starting index
        total_len = len(st.session_state.simulator.data)
        start_idx = total_len - max_points
        
        # Stream data
        for data_point in st.session_state.simulator.stream_data(
            start_index=start_idx,
            max_points=max_points
        ):
            if not st.session_state.is_streaming:
                break
            
            # Make prediction
            exog_data = {
                'weather_cluster': data_point['weather_cluster'],
                'holiday': data_point['holiday']
            }
            prediction = st.session_state.predictor.predict_next(exog_data)
            
            # Store results
            st.session_state.data_stream.append({
                'date': data_point['date'],
                'actual': data_point['actual_energy'],
                'predicted': prediction,
                'temperature': data_point['temperature'],
                'humidity': data_point['humidity'],
                'wind_speed': data_point['wind_speed']
            })
            
            # Update history
            st.session_state.predictor.update_history(
                actual=data_point['actual_energy'],
                predicted=prediction
            )
            
            # Update visualization
            df = pd.DataFrame(st.session_state.data_stream)
            fig = create_real_time_chart(df)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update status
            status_placeholder.info(
                f"Streaming... Point {len(st.session_state.data_stream)}/{max_points} | "
                f"Date: {data_point['date']} | "
                f"Actual: {data_point['actual_energy']:.2f} kWh | "
                f"Predicted: {prediction:.2f} kWh"
            )
        
        # Stream completed
        st.session_state.is_streaming = False
        status_placeholder.success("Streaming completed!")
    
    elif st.session_state.stream_started and len(st.session_state.data_stream) > 0:
        # Show last state
        df = pd.DataFrame(st.session_state.data_stream)
        fig = create_real_time_chart(df)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        status_placeholder.success("Stream paused. Click 'Start Stream' to continue.")
    
    # Data table
    if len(st.session_state.data_stream) > 0:
        st.subheader("Recent Predictions")
        df_display = pd.DataFrame(st.session_state.data_stream).tail(10)
        df_display['error'] = df_display['actual'] - df_display['predicted']
        df_display['error_pct'] = (df_display['error'] / df_display['actual'] * 100).abs()
        st.dataframe(
            df_display[['date', 'actual', 'predicted', 'error', 'error_pct', 'temperature']].style.format({
                'actual': '{:.2f}',
                'predicted': '{:.2f}',
                'error': '{:.2f}',
                'error_pct': '{:.1f}%',
                'temperature': '{:.1f}°C'
            }),
            use_container_width=True
        )


if __name__ == "__main__":
    main()
