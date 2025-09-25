import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Additional imports for new models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    st.success("✅ TensorFlow available - LSTM model enabled")
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    st.warning(f"⚠️ TensorFlow not available: {str(e)[:100]}... LSTM model will be disabled.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
    st.success("✅ Statsmodels available - ARIMA model enabled")
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    st.warning(f"⚠️ Statsmodels not available: {str(e)}. ARIMA model will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    st.success("✅ Prophet available - Prophet model enabled")
except ImportError as e:
    PROPHET_AVAILABLE = False
    st.warning(f"⚠️ Prophet not available: {str(e)}. Prophet model will be disabled.")

# Set page configuration
st.set_page_config(
    page_title="Energy Data Dashboard", 
    page_icon="⚡", 
    layout="wide"
)

st.title("⚡ Energy Data Dashboard")
st.markdown("Interactive visualization of wind, solar, and pricing data with time period selection")

# Model availability status
st.subheader("🤖 Model Availability Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if TENSORFLOW_AVAILABLE:
        st.success("🟢 LSTM Available")
    else:
        st.error("🔴 LSTM Unavailable")

with col2:
    if STATSMODELS_AVAILABLE:
        st.success("🟢 ARIMA Available")
    else:
        st.error("🔴 ARIMA Unavailable")

with col3:
    if PROPHET_AVAILABLE:
        st.success("🟢 Prophet Available")
    else:
        st.error("🔴 Prophet Unavailable")

with col4:
    st.success("🟢 Random Forest Available")

# Troubleshooting section
if not TENSORFLOW_AVAILABLE:
    with st.expander("🔧 TensorFlow Troubleshooting", expanded=False):
        st.markdown("""
        **TensorFlow Installation Issues:**
        
        If you're seeing TensorFlow errors, try these solutions:
        
        1. **Reinstall TensorFlow:**
           ```bash
           pip uninstall tensorflow
           pip install tensorflow
           ```
        
        2. **Try TensorFlow CPU version:**
           ```bash
           pip install tensorflow-cpu
           ```
        
        3. **Check Python version compatibility:**
           - TensorFlow requires Python 3.8-3.11
           - Your current version: Check with `python --version`
        
        4. **Alternative: Use other models**
           - ARIMA and Prophet models are still available
           - Random Forest works without TensorFlow
        """)

# Show available models count
available_count = sum([TENSORFLOW_AVAILABLE, STATSMODELS_AVAILABLE, PROPHET_AVAILABLE, True])  # +1 for Random Forest
st.info(f"📊 **Available Models**: {available_count}/4 models ready for prediction")

@st.cache_data
def load_data():
    """Load the real energy data from CSV file"""
    try:
        # Load the CSV file with semicolon separator
        df = pd.read_csv("combined_energy_price_clean.csv", sep=';')
        
        # Display column names for debugging (only show if there are issues)
        # st.write("📋 Detected columns:", list(df.columns))
        
        # Convert Datetime column to datetime type
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Clean numeric columns - handle any formatting issues
        numeric_columns = ['Total_Wind', 'Total_Solar', 'Price_MWh']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, forcing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values in critical columns
        df = df.dropna(subset=['Datetime', 'Total_Wind', 'Total_Solar', 'Price_MWh'])
        
        # Data quality checks and cleaning
        initial_rows = len(df)
        
        # Remove unrealistic values (likely data errors)
        df = df[df['Total_Wind'] >= 0]  # Wind can't be negative
        df = df[df['Total_Solar'] >= 0]  # Solar can't be negative  
        df = df[df['Price_MWh'] >= 0]   # Price can't be negative
        df = df[df['Price_MWh'] <= 1000]  # Remove extreme price outliers
        df = df[df['Total_Wind'] <= 50000]  # Remove extreme wind outliers
        df = df[df['Total_Solar'] <= 50000]  # Remove extreme solar outliers
        
        cleaned_rows = len(df)
        if initial_rows != cleaned_rows:
            st.info(f"🧹 Cleaned data: Removed {initial_rows - cleaned_rows:,} rows with unrealistic values")
        
        # Extract additional time-based features if not already present
        if 'Hour' not in df.columns:
            df['Hour'] = df['Datetime'].dt.hour
        if 'DayOfWeek' not in df.columns:
            df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        if 'Month' not in df.columns:
            df['Month'] = df['Datetime'].dt.month
        
        # Sort by datetime to ensure proper ordering
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        st.error("❌ File 'combined_energy_price_clean.csv' not found!")
        st.info("Please make sure the CSV file is in the same directory as your Streamlit app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check that your CSV file has the expected format.")
        
        # Show file preview for debugging
        try:
            with open("combined_energy_price_clean.csv", 'r') as f:
                preview = f.read(500)
                st.text("File preview (first 500 characters):")
                st.code(preview)
        except:
            pass
        st.stop()

# Load data
with st.spinner("Loading your energy data..."):
    df = load_data()

# Remove the debug column display since it's working now
if len(df) == 0:
    st.error("❌ No valid data remaining after cleaning. Please check your data file.")
    st.stop()

# Prediction functions
def create_features(df, target_col, lookback_hours=24, is_prediction=False):
    """Create features for time series prediction with enhanced feature engineering"""
    features_df = df.copy()
    
    # Enhanced time-based features
    features_df['hour'] = features_df['Datetime'].dt.hour
    features_df['day_of_week'] = features_df['Datetime'].dt.dayofweek
    features_df['month'] = features_df['Datetime'].dt.month
    features_df['day_of_year'] = features_df['Datetime'].dt.dayofyear
    features_df['quarter'] = features_df['Datetime'].dt.quarter
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    
    # Enhanced cyclical encoding for time features
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
    features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
    features_df['day_of_year_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['day_of_year_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    
    # Demand pattern features (peak hours, business hours, etc.)
    features_df['is_peak_hour'] = ((features_df['hour'] >= 8) & (features_df['hour'] <= 20)).astype(int)
    features_df['is_business_hour'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)
    features_df['is_night'] = ((features_df['hour'] >= 22) | (features_df['hour'] <= 6)).astype(int)
    
    # Lag features (only for training, not prediction)
    if not is_prediction:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            if lag < len(features_df):
                features_df[f'{target_col}_lag_{lag}'] = features_df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24, 48, 168]:  # Added weekly window
            if window < len(features_df):
                features_df[f'{target_col}_rolling_mean_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).mean()
                features_df[f'{target_col}_rolling_std_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
                features_df[f'{target_col}_rolling_max_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).max()
                features_df[f'{target_col}_rolling_min_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).min()
    
    # Enhanced cross-feature interactions for price prediction
    if target_col == 'Price_MWh':
        features_df['total_generation'] = features_df['Total_Wind'] + features_df['Total_Solar']
        features_df['renewable_ratio'] = features_df['Total_Solar'] / (features_df['Total_Wind'] + features_df['Total_Solar'] + 1)
        features_df['wind_solar_ratio'] = features_df['Total_Wind'] / (features_df['Total_Solar'] + 1)
        
        # Price volatility features
        if not is_prediction:
            features_df['price_volatility_24h'] = features_df['Price_MWh'].rolling(window=24, min_periods=1).std().fillna(0)
            features_df['price_trend_24h'] = (features_df['Price_MWh'] - features_df['Price_MWh'].shift(24)).fillna(0)
        
        # Generation patterns
        features_df['solar_availability'] = (features_df['Total_Solar'] > 1000).astype(int)  # High solar generation
        features_df['wind_availability'] = (features_df['Total_Wind'] > 1000).astype(int)  # High wind generation
    
    return features_df

def train_prediction_model(df, target_col, test_size=0.2):
    """Train a prediction model for the target column"""
    
    # Create features
    features_df = create_features(df, target_col)
    
    # Remove rows with NaN values (due to lag features)
    features_df = features_df.dropna()
    
    if len(features_df) < 100:
        return None, None, None
    
    # Additional data quality checks for price prediction
    if target_col == 'Price_MWh':
        # Check if we have realistic price data
        price_data = features_df[target_col]
        if price_data.mean() < 20 or price_data.mean() > 200:
            st.warning(f"⚠️ Unusual price data detected (mean: ${price_data.mean():.2f}/MWh). Model may not perform well.")
        
        # Remove extreme outliers that could confuse the model
        price_q1 = price_data.quantile(0.25)
        price_q3 = price_data.quantile(0.75)
        price_iqr = price_q3 - price_q1
        price_lower_bound = price_q1 - 3 * price_iqr
        price_upper_bound = price_q3 + 3 * price_iqr
        
        # Keep only reasonable price data
        features_df = features_df[
            (features_df[target_col] >= max(0, price_lower_bound)) & 
            (features_df[target_col] <= min(1000, price_upper_bound))
        ]
        
        if len(features_df) < 50:
            st.error("❌ Not enough valid price data after outlier removal")
            return None, None, None
    
    # Select feature columns (exclude datetime and target)
    feature_cols = [col for col in features_df.columns 
                   if col not in ['Datetime', target_col] and not col.startswith('Total_')]
    
    X = features_df[feature_cols]
    y = features_df[target_col]
    
    # Split data chronologically (latest data for testing)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with improved parameters
    model = RandomForestRegressor(
        n_estimators=200,  # More trees for better performance
        max_depth=20,     # Deeper trees for complex patterns
        min_samples_split=5,  # Allow more splits
        min_samples_leaf=2,   # Allow smaller leaves
        max_features='sqrt',  # Feature subsampling
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate feature importance
    feature_importance = model.feature_importances_
    feature_names = feature_cols
    importance_dict = dict(zip(feature_names, feature_importance))
    
    # Calculate R-squared
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, {
        'mae': mae, 
        'rmse': rmse, 
        'r2': r2,
        'feature_cols': feature_cols,
        'feature_importance': importance_dict
    }

def train_lstm_model(df, target_col, sequence_length=24, test_size=0.2):
    """Train an LSTM model for time series prediction"""
    if not TENSORFLOW_AVAILABLE:
        return None, None, None
    
    try:
        # Prepare data
        data = df[target_col].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(scaled_data, sequence_length)
        
        if len(X) < 100:
            return None, None, None
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return model, scaler, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'sequence_length': sequence_length
        }
    except Exception as e:
        st.error(f"LSTM training failed: {str(e)}")
        return None, None, None

def train_arima_model(df, target_col, test_size=0.2):
    """Train an ARIMA model for time series prediction"""
    if not STATSMODELS_AVAILABLE:
        return None, None, None
    
    try:
        data = df[target_col].values
        
        # Split data
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        if len(train_data) < 50:
            return None, None, None
        
        # Auto ARIMA parameter selection (simplified)
        # Try different combinations
        best_aic = float('inf')
        best_model = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            return None, None, None
        
        # Make predictions
        forecast = best_model.forecast(steps=len(test_data))
        
        # Evaluate
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        r2 = r2_score(test_data, forecast)
        
        return best_model, None, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'aic': best_aic
        }
    except Exception as e:
        st.error(f"ARIMA training failed: {str(e)}")
        return None, None, None

def train_prophet_model(df, target_col, test_size=0.2):
    """Train a Prophet model for time series prediction"""
    if not PROPHET_AVAILABLE:
        return None, None, None
    
    try:
        # Prepare data for Prophet
        prophet_df = df[['Datetime', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Split data
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        if len(train_df) < 50:
            return None, None, None
        
        # Create and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        model.fit(train_df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test_df), freq='H')
        forecast = model.predict(future)
        
        # Get predictions for test period
        test_predictions = forecast['yhat'].iloc[split_idx:].values
        test_actual = test_df['y'].values
        
        # Evaluate
        mae = mean_absolute_error(test_actual, test_predictions)
        rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
        r2 = r2_score(test_actual, test_predictions)
        
        return model, None, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    except Exception as e:
        st.error(f"Prophet training failed: {str(e)}")
        return None, None, None

def make_predictions(model, scaler, df, target_col, hours_ahead=24, feature_cols=None):
    """Make predictions for the next N hours with improved feature handling"""
    
    # Get the latest data point
    latest_data = df.iloc[-hours_ahead*3:].copy()  # Use more data for better context
    
    predictions = []
    current_data = latest_data.copy()
    
    # Get the last datetime and generate future timestamps
    last_datetime = current_data['Datetime'].iloc[-1]
    future_datetimes = pd.date_range(
        start=last_datetime + timedelta(hours=1),
        periods=hours_ahead,
        freq='H'
    )
    
    # If feature_cols not provided, get them from current data
    if feature_cols is None:
        training_features = create_features(current_data, target_col, is_prediction=True)
        feature_cols = [col for col in training_features.columns 
                       if col not in ['Datetime', target_col] and not col.startswith('Total_')]
    
    for i, future_datetime in enumerate(future_datetimes):
        # Create a future data point with proper time features
        future_row = current_data.iloc[-1:].copy()
        future_row['Datetime'] = future_datetime
        
        # For solar predictions, use realistic solar patterns
        if target_col == 'Total_Solar':
            # Solar follows daily patterns - use historical average for this hour
            hour_avg = current_data[current_data['Datetime'].dt.hour == future_datetime.hour]['Total_Solar'].mean()
            if pd.isna(hour_avg):
                hour_avg = current_data['Total_Solar'].mean()
            future_row['Total_Solar'] = max(0, hour_avg * (0.8 + 0.4 * np.random.random()))  # Add some variation
        
        # For wind, use recent average with some variation
        if target_col == 'Total_Wind':
            recent_wind = current_data['Total_Wind'].tail(24).mean()
            future_row['Total_Wind'] = max(0, recent_wind * (0.7 + 0.6 * np.random.random()))
        
        # Add the future row to current data
        current_data = pd.concat([current_data, future_row], ignore_index=True)
        
        # Create features for this future point
        features_df = create_features(current_data, target_col, is_prediction=True)
        
        if len(features_df) == 0:
            break
            
        # Create a feature row with all required columns
        latest_features = pd.DataFrame(index=[0])
        
        # Add all required features, using 0 for missing ones
        for col in feature_cols:
            if col in features_df.columns:
                latest_features[col] = features_df[col].iloc[-1] if not pd.isna(features_df[col].iloc[-1]) else 0
            else:
                latest_features[col] = 0
        
        # Fill any NaN values
        latest_features = latest_features.fillna(0)
        
        # Scale features
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make prediction
        pred = model.predict(latest_features_scaled)[0]
        
        # Apply realistic constraints with better price validation
        if target_col == 'Price_MWh':
            # Get recent price statistics for realistic bounds
            recent_prices = current_data['Price_MWh'].tail(168)  # Last week
            price_mean = recent_prices.mean()
            price_std = recent_prices.std()
            
            # Set realistic price bounds based on historical data
            min_price = max(10, price_mean - 2 * price_std)  # Minimum $10/MWh
            max_price = min(500, price_mean + 3 * price_std)  # Maximum $500/MWh
            
            # Apply constraints
            pred = max(min_price, pred)  # Ensure minimum realistic price
            pred = min(max_price, pred)  # Cap at reasonable maximum
            
            # Additional validation: if prediction is too low, use recent average
            if pred < price_mean * 0.3:  # If prediction is less than 30% of recent average
                pred = price_mean * (0.8 + 0.4 * np.random.random())  # Use recent average with variation
                
        elif target_col == 'Total_Solar':
            pred = max(0, pred)  # Solar can't be negative
            # Solar should follow daily patterns - add some validation
            if future_datetime.hour < 6 or future_datetime.hour > 20:  # Night hours
                pred = min(pred, 100)  # Very low solar at night
        elif target_col == 'Total_Wind':
            pred = max(0, pred)  # Wind can't be negative
        
        predictions.append(pred)
        
        # Update the future row with the predicted value
        current_data.loc[current_data['Datetime'] == future_datetime, target_col] = pred
    
    return future_datetimes, predictions

def make_lstm_predictions(model, scaler, df, target_col, hours_ahead=24):
    """Make predictions using LSTM model"""
    if not TENSORFLOW_AVAILABLE or model is None:
        return None, None
    
    try:
        # Get recent data
        recent_data = df[target_col].tail(24).values.reshape(-1, 1)
        scaled_data = scaler.transform(recent_data)
        
        predictions = []
        current_sequence = scaled_data[-24:].reshape(1, 24, 1)
        
        for _ in range(hours_ahead):
            # Make prediction
            pred_scaled = model.predict(current_sequence, verbose=0)[0][0]
            pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]
            
            # Apply constraints
            if target_col == 'Price_MWh':
                pred_original = max(10, min(500, pred_original))
            elif target_col in ['Total_Solar', 'Total_Wind']:
                pred_original = max(0, pred_original)
            
            predictions.append(pred_original)
            
            # Update sequence for next prediction
            new_value = np.array([[pred_scaled]])
            current_sequence = np.concatenate([current_sequence[:, 1:, :], new_value.reshape(1, 1, 1)], axis=1)
        
        # Generate future timestamps
        last_datetime = df['Datetime'].iloc[-1]
        future_datetimes = pd.date_range(
            start=last_datetime + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        return future_datetimes, predictions
    except Exception as e:
        st.error(f"LSTM prediction failed: {str(e)}")
        return None, None

def make_arima_predictions(model, scaler, df, target_col, hours_ahead=24):
    """Make predictions using ARIMA model"""
    if not STATSMODELS_AVAILABLE or model is None:
        return None, None
    
    try:
        # Make predictions
        forecast = model.forecast(steps=hours_ahead)
        
        # Apply constraints
        predictions = []
        for pred in forecast:
            if target_col == 'Price_MWh':
                pred = max(10, min(500, pred))
            elif target_col in ['Total_Solar', 'Total_Wind']:
                pred = max(0, pred)
            predictions.append(pred)
        
        # Generate future timestamps
        last_datetime = df['Datetime'].iloc[-1]
        future_datetimes = pd.date_range(
            start=last_datetime + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        return future_datetimes, predictions
    except Exception as e:
        st.error(f"ARIMA prediction failed: {str(e)}")
        return None, None

def make_prophet_predictions(model, scaler, df, target_col, hours_ahead=24):
    """Make predictions using Prophet model"""
    if not PROPHET_AVAILABLE or model is None:
        return None, None
    
    try:
        # Create future dataframe
        last_datetime = df['Datetime'].iloc[-1]
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=last_datetime + timedelta(hours=1),
                periods=hours_ahead,
                freq='H'
            )
        })
        
        # Make predictions
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
        
        # Apply constraints
        for i, pred in enumerate(predictions):
            if target_col == 'Price_MWh':
                predictions[i] = max(10, min(500, pred))
            elif target_col in ['Total_Solar', 'Total_Wind']:
                predictions[i] = max(0, pred)
        
        return future['ds'], predictions
    except Exception as e:
        st.error(f"Prophet prediction failed: {str(e)}")
        return None, None

st.success(f"Loaded {len(df):,} data points from {df['Datetime'].min()} to {df['Datetime'].max()}")

# Sidebar for time period selection
st.sidebar.header("📅 Time Period Selection")

# Quick preset options
preset_option = st.sidebar.selectbox(
    "Quick Select:",
    ["Custom Range", "Last 7 Days", "Last 30 Days", "Last 90 Days", 
     "Last Year", "Full Dataset", "2024 Data", "2023 Data"]
)

# Calculate date ranges based on preset
max_date = df['Datetime'].max()
min_date = df['Datetime'].min()

if preset_option == "Last 7 Days":
    start_date = max_date - timedelta(days=7)
    end_date = max_date
elif preset_option == "Last 30 Days":
    start_date = max_date - timedelta(days=30)
    end_date = max_date
elif preset_option == "Last 90 Days":
    start_date = max_date - timedelta(days=90)
    end_date = max_date
elif preset_option == "Last Year":
    start_date = max_date - timedelta(days=365)
    end_date = max_date
elif preset_option == "Full Dataset":
    start_date = min_date
    end_date = max_date
elif preset_option == "2024 Data":
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31, 23)
elif preset_option == "2023 Data":
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31, 23)
else:  # Custom Range
    start_date = st.sidebar.date_input(
        "Start Date:",
        value=max_date.date() - timedelta(days=30),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    end_date = st.sidebar.date_input(
        "End Date:",
        value=max_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Convert to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

# Filter data based on selected period
filtered_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)].copy()

if filtered_df.empty:
    st.error("No data available for the selected time period.")
    st.stop()

# Display summary statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📊 Data Points", 
        f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df):,} vs full dataset"
    )

with col2:
    avg_wind = filtered_df['Total_Wind'].mean()
    st.metric(
        "💨 Avg Wind", 
        f"{avg_wind:.1f} MW"
    )

with col3:
    avg_solar = filtered_df['Total_Solar'].mean()
    st.metric(
        "☀️ Avg Solar", 
        f"{avg_solar:.1f} MW"
    )

with col4:
    avg_price = filtered_df['Price_MWh'].mean()
    st.metric(
        "💰 Avg Price", 
        f"${avg_price:.2f}/MWh"
    )

# Prediction settings
st.sidebar.header("🔮 Prediction Settings")
enable_predictions = st.sidebar.checkbox("Enable Predictions", False)
if enable_predictions:
    prediction_hours = st.sidebar.selectbox(
        "Prediction Horizon:",
        [24, 48],
        index=0
    )
    predict_price = st.sidebar.checkbox("Predict Price", True)
    predict_solar = st.sidebar.checkbox("Predict Solar", True)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    available_models = ["Random Forest"]
    
    if TENSORFLOW_AVAILABLE:
        available_models.append("LSTM")
    if STATSMODELS_AVAILABLE:
        available_models.append("ARIMA")
    if PROPHET_AVAILABLE:
        available_models.append("Prophet")
    
    selected_models = st.sidebar.multiselect(
        "Choose Models to Compare:",
        available_models,
        default=["Random Forest"] if "Random Forest" in available_models else available_models[:1]
    )

# Prediction section
predictions_data = {}
if enable_predictions and selected_models:
    if predict_price or predict_solar:
        st.subheader("🔮 Training Multiple Prediction Models")
        
        # Use recent data for training (last 2 years for better performance)
        training_cutoff = filtered_df['Datetime'].max() - timedelta(days=730)
        training_data = df[df['Datetime'] >= training_cutoff].copy()
        
        total_models = len(selected_models) * (2 if predict_price and predict_solar else 1)
        current_model = 0
        
        # Train models for price prediction
        if predict_price:
            st.subheader("💰 Price Prediction Models")
            price_models = {}
            
            for model_name in selected_models:
                with st.spinner(f"Training {model_name} for price prediction..."):
                    if model_name == "Random Forest":
                        model, scaler, metrics = train_prediction_model(training_data, 'Price_MWh')
                        if model:
                            future_times, preds = make_predictions(
                                model, scaler, training_data, 'Price_MWh', prediction_hours, metrics['feature_cols']
                            )
                            price_models[model_name] = {
                                'times': future_times,
                                'values': preds,
                                'metrics': metrics,
                                'color': '#2ca02c'  # Green
                            }
                            st.success(f"✅ {model_name} - MAE: {metrics['mae']:.2f} $/MWh, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "LSTM":
                        model, scaler, metrics = train_lstm_model(training_data, 'Price_MWh')
                        if model:
                            future_times, preds = make_lstm_predictions(model, scaler, training_data, 'Price_MWh', prediction_hours)
                            if future_times is not None:
                                price_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#ff7f0e'  # Orange
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.2f} $/MWh, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "ARIMA":
                        model, scaler, metrics = train_arima_model(training_data, 'Price_MWh')
                        if model:
                            future_times, preds = make_arima_predictions(model, scaler, training_data, 'Price_MWh', prediction_hours)
                            if future_times is not None:
                                price_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#1f77b4'  # Blue
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.2f} $/MWh, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "Prophet":
                        model, scaler, metrics = train_prophet_model(training_data, 'Price_MWh')
                        if model:
                            future_times, preds = make_prophet_predictions(model, scaler, training_data, 'Price_MWh', prediction_hours)
                            if future_times is not None:
                                price_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#9467bd'  # Purple
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.2f} $/MWh, R²: {metrics['r2']:.3f}")
                
                current_model += 1
                progress = current_model / total_models
                st.progress(progress)
            
            if price_models:
                predictions_data['price'] = price_models
        
        # Train models for solar prediction
        if predict_solar:
            st.subheader("☀️ Solar Prediction Models")
            solar_models = {}
            
            for model_name in selected_models:
                with st.spinner(f"Training {model_name} for solar prediction..."):
                    if model_name == "Random Forest":
                        model, scaler, metrics = train_prediction_model(training_data, 'Total_Solar')
                        if model:
                            future_times, preds = make_predictions(
                                model, scaler, training_data, 'Total_Solar', prediction_hours, metrics['feature_cols']
                            )
                            solar_models[model_name] = {
                                'times': future_times,
                                'values': preds,
                                'metrics': metrics,
                                'color': '#ff7f0e'  # Orange
                            }
                            st.success(f"✅ {model_name} - MAE: {metrics['mae']:.1f} MW, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "LSTM":
                        model, scaler, metrics = train_lstm_model(training_data, 'Total_Solar')
                        if model:
                            future_times, preds = make_lstm_predictions(model, scaler, training_data, 'Total_Solar', prediction_hours)
                            if future_times is not None:
                                solar_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#2ca02c'  # Green
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.1f} MW, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "ARIMA":
                        model, scaler, metrics = train_arima_model(training_data, 'Total_Solar')
                        if model:
                            future_times, preds = make_arima_predictions(model, scaler, training_data, 'Total_Solar', prediction_hours)
                            if future_times is not None:
                                solar_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#1f77b4'  # Blue
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.1f} MW, R²: {metrics['r2']:.3f}")
                    
                    elif model_name == "Prophet":
                        model, scaler, metrics = train_prophet_model(training_data, 'Total_Solar')
                        if model:
                            future_times, preds = make_prophet_predictions(model, scaler, training_data, 'Total_Solar', prediction_hours)
                            if future_times is not None:
                                solar_models[model_name] = {
                                    'times': future_times,
                                    'values': preds,
                                    'metrics': metrics,
                                    'color': '#9467bd'  # Purple
                                }
                                st.success(f"✅ {model_name} - MAE: {metrics['mae']:.1f} MW, R²: {metrics['r2']:.3f}")
                
                current_model += 1
                progress = current_model / total_models
                st.progress(progress)
            
            if solar_models:
                predictions_data['solar'] = solar_models

# Chart options
st.sidebar.header("📈 Chart Options")
chart_type = st.sidebar.selectbox(
    "Chart Type:",
    ["Time Series", "Daily Patterns", "Monthly Patterns", "Correlation Analysis"]
)

show_wind = st.sidebar.checkbox("Show Wind Data", True)
show_solar = st.sidebar.checkbox("Show Solar Data", True)
show_price = st.sidebar.checkbox("Show Price Data", True)

# Create visualizations based on chart type
if chart_type == "Time Series":
    # Create subplots with separate wind and solar panels for better visibility
    subplot_titles = []
    if show_wind:
        subplot_titles.append('Wind Generation (MW)')
    if show_solar:
        subplot_titles.append('Solar Generation (MW)')
    if show_price:
        subplot_titles.append('Price ($/MWh)')
    
    num_subplots = len(subplot_titles)
    if num_subplots == 0:
        st.warning("Please select at least one data type to display.")
        st.stop()
    
    fig = make_subplots(
        rows=num_subplots, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    current_row = 1
    
    # Add wind data in its own subplot
    if show_wind:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Datetime'], 
                y=filtered_df['Total_Wind'],
                name='Wind', 
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Wind</b><br>Date: %{x}<br>Power: %{y:.1f} MW<extra></extra>'
            ), 
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Wind Power (MW)", row=current_row, col=1)
        current_row += 1
    
    # Add solar data in its own subplot
    if show_solar:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Datetime'], 
                y=filtered_df['Total_Solar'],
                name='Solar (Historical)', 
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Solar</b><br>Date: %{x}<br>Power: %{y:.1f} MW<extra></extra>'
            ), 
            row=current_row, col=1
        )
        
        # Add solar predictions if available (multiple models)
        if enable_predictions and 'solar' in predictions_data:
            for model_name, model_data in predictions_data['solar'].items():
                fig.add_trace(
                    go.Scatter(
                        x=model_data['times'], 
                        y=model_data['values'],
                        name=f'Solar ({model_name})', 
                        line=dict(color=model_data['color'], width=3, dash='dash'),
                        hovertemplate=f'<b>Solar {model_name}</b><br>Date: %{{x}}<br>Power: %{{y:.1f}} MW<extra></extra>'
                    ), 
                    row=current_row, col=1
                )
        
        fig.update_yaxes(title_text="Solar Power (MW)", row=current_row, col=1)
        current_row += 1
    
    # Add price data in its own subplot
    if show_price:
        fig.add_trace(
            go.Scatter(
                x=filtered_df['Datetime'], 
                y=filtered_df['Price_MWh'],
                name='Price (Historical)', 
                line=dict(color='#2ca02c', width=2),
                hovertemplate='<b>Price</b><br>Date: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
            ), 
            row=current_row, col=1
        )
        
        # Add price predictions if available (multiple models)
        if enable_predictions and 'price' in predictions_data:
            for model_name, model_data in predictions_data['price'].items():
                fig.add_trace(
                    go.Scatter(
                        x=model_data['times'], 
                        y=model_data['values'],
                        name=f'Price ({model_name})', 
                        line=dict(color=model_data['color'], width=3, dash='dash'),
                        hovertemplate=f'<b>Price {model_name}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}/MWh<extra></extra>'
                    ), 
                    row=current_row, col=1
                )
        
        fig.update_yaxes(title_text="Price ($/MWh)", row=current_row, col=1)
    
    # Update layout
    chart_title = f"Energy Data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    if enable_predictions and predictions_data:
        chart_title += f" (+ {prediction_hours}h Predictions)"
    
    fig.update_layout(
        height=200 * num_subplots + 100,  # Dynamic height based on number of subplots
        title_text=chart_title,
        title_x=0.5,
        hovermode='x unified',
        showlegend=True
    )

elif chart_type == "Daily Patterns":
    # Group by hour to show daily patterns
    daily_pattern = filtered_df.groupby('Hour').agg({
        'Total_Wind': 'mean',
        'Total_Solar': 'mean',
        'Price_MWh': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    if show_wind:
        fig.add_trace(go.Scatter(
            x=daily_pattern['Hour'],
            y=daily_pattern['Total_Wind'],
            name='Avg Wind',
            line=dict(color='#1f77b4'),
            mode='lines+markers'
        ))
    
    if show_solar:
        fig.add_trace(go.Scatter(
            x=daily_pattern['Hour'],
            y=daily_pattern['Total_Solar'],
            name='Avg Solar',
            line=dict(color='#ff7f0e'),
            mode='lines+markers'
        ))
    
    if show_price:
        fig.add_trace(go.Scatter(
            x=daily_pattern['Hour'],
            y=daily_pattern['Price_MWh'],
            name='Avg Price',
            line=dict(color='#2ca02c'),
            mode='lines+markers',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Average Daily Patterns",
        xaxis_title="Hour of Day",
        yaxis_title="Power (MW)",
        yaxis2=dict(
            title="Price ($/MWh)",
            overlaying="y",
            side="right"
        ),
        height=500
    )

elif chart_type == "Monthly Patterns":
    # Group by month to show seasonal patterns
    monthly_pattern = filtered_df.groupby('Month').agg({
        'Total_Wind': 'mean',
        'Total_Solar': 'mean',
        'Price_MWh': 'mean'
    }).reset_index()
    
    # Map month numbers to names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pattern['Month_Name'] = monthly_pattern['Month'].map(
        lambda x: month_names[x-1]
    )
    
    fig = go.Figure()
    
    if show_wind:
        fig.add_trace(go.Bar(
            x=monthly_pattern['Month_Name'],
            y=monthly_pattern['Total_Wind'],
            name='Avg Wind',
            marker_color='#1f77b4'
        ))
    
    if show_solar:
        fig.add_trace(go.Bar(
            x=monthly_pattern['Month_Name'],
            y=monthly_pattern['Total_Solar'],
            name='Avg Solar',
            marker_color='#ff7f0e'
        ))
    
    if show_price:
        fig.add_trace(go.Scatter(
            x=monthly_pattern['Month_Name'],
            y=monthly_pattern['Price_MWh'],
            name='Avg Price',
            line=dict(color='#2ca02c'),
            mode='lines+markers',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Average Monthly Patterns",
        xaxis_title="Month",
        yaxis_title="Power (MW)",
        yaxis2=dict(
            title="Price ($/MWh)",
            overlaying="y",
            side="right"
        ),
        height=500
    )

else:  # Correlation Analysis
    # Create correlation scatter plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wind vs Price', 'Solar vs Price'),
        horizontal_spacing=0.1
    )
    
    # Sample data for performance if dataset is large
    sample_df = filtered_df.sample(n=min(5000, len(filtered_df)))
    
    if show_wind and show_price:
        fig.add_trace(
            go.Scatter(
                x=sample_df['Total_Wind'],
                y=sample_df['Price_MWh'],
                mode='markers',
                name='Wind vs Price',
                marker=dict(color='#1f77b4', opacity=0.6),
                hovertemplate='Wind: %{x:.1f} MW<br>Price: $%{y:.2f}/MWh<extra></extra>'
            ),
            row=1, col=1
        )
    
    if show_solar and show_price:
        fig.add_trace(
            go.Scatter(
                x=sample_df['Total_Solar'],
                y=sample_df['Price_MWh'],
                mode='markers',
                name='Solar vs Price',
                marker=dict(color='#ff7f0e', opacity=0.6),
                hovertemplate='Solar: %{x:.1f} MW<br>Price: $%{y:.2f}/MWh<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="Energy Generation vs Price Correlation",
        height=500
    )
    
    fig.update_xaxes(title_text="Wind Power (MW)", row=1, col=1)
    fig.update_xaxes(title_text="Solar Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($/MWh)", row=1, col=2)

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Show prediction metrics if available
if enable_predictions and predictions_data:
    st.subheader("📊 Model Performance Comparison")
    
    # Create model comparison tables
    if 'price' in predictions_data:
        st.subheader("💰 Price Prediction Models")
        
        # Create comparison table
        price_comparison_data = []
        for model_name, model_data in predictions_data['price'].items():
            price_comparison_data.append({
                'Model': model_name,
                'MAE ($/MWh)': f"{model_data['metrics']['mae']:.2f}",
                'RMSE ($/MWh)': f"{model_data['metrics']['rmse']:.2f}",
                'R²': f"{model_data['metrics']['r2']:.3f}",
                'Color': model_data['color']
            })
        
        price_df = pd.DataFrame(price_comparison_data)
        st.dataframe(price_df, use_container_width=True)
        
        # Show best model
        best_price_model = min(predictions_data['price'].items(), 
                              key=lambda x: x[1]['metrics']['mae'])
        st.success(f"🏆 **Best Price Model**: {best_price_model[0]} (MAE: ${best_price_model[1]['metrics']['mae']:.2f}/MWh)")
    
    if 'solar' in predictions_data:
        st.subheader("☀️ Solar Prediction Models")
        
        # Create comparison table
        solar_comparison_data = []
        for model_name, model_data in predictions_data['solar'].items():
            solar_comparison_data.append({
                'Model': model_name,
                'MAE (MW)': f"{model_data['metrics']['mae']:.1f}",
                'RMSE (MW)': f"{model_data['metrics']['rmse']:.1f}",
                'R²': f"{model_data['metrics']['r2']:.3f}",
                'Color': model_data['color']
            })
        
        solar_df = pd.DataFrame(solar_comparison_data)
        st.dataframe(solar_df, use_container_width=True)
        
        # Show best model
        best_solar_model = min(predictions_data['solar'].items(), 
                              key=lambda x: x[1]['metrics']['mae'])
        st.success(f"🏆 **Best Solar Model**: {best_solar_model[0]} (MAE: {best_solar_model[1]['metrics']['mae']:.1f} MW)")
    
    # Feature importance analysis
    if st.checkbox("🔍 Show Feature Importance Analysis"):
        st.subheader("🎯 Most Important Features")
        
        for pred_type, pred_data in predictions_data.items():
            if 'feature_importance' in pred_data['metrics']:
                st.write(f"**{pred_type.title()} Model Feature Importance:**")
                
                # Get top 10 most important features
                importance = pred_data['metrics']['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Create a simple bar chart
                import plotly.express as px
                feature_names = [f[0] for f in sorted_features]
                importance_values = [f[1] for f in sorted_features]
                
                fig_importance = px.bar(
                    x=importance_values,
                    y=feature_names,
                    orientation='h',
                    title=f"Top 10 Features for {pred_type.title()} Prediction",
                    labels={'x': 'Importance Score', 'y': 'Feature'}
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Show next few prediction values with validation
    st.subheader("🔮 Next Few Hours Predictions")
    
    # Add price validation info
    if 'price' in predictions_data:
        recent_prices = df['Price_MWh'].tail(168)  # Last week
        price_mean = recent_prices.mean()
        price_std = recent_prices.std()
        
        st.info(f"📊 **Price Context**: Recent average: ${price_mean:.2f}/MWh (std: ${price_std:.2f})")
    
    # Display predictions for each model
    for pred_type, models_data in predictions_data.items():
        st.subheader(f"🔮 {pred_type.title()} Predictions by Model")
        
        for model_name, model_data in models_data.items():
            with st.expander(f"📊 {model_name} Model Predictions", expanded=True):
                st.write(f"**Model Performance**: MAE: {model_data['metrics']['mae']:.2f}, R²: {model_data['metrics']['r2']:.3f}")
                
                # Show first 6 predictions
                for j, (time, value) in enumerate(zip(model_data['times'][:6], model_data['values'][:6])):
                    if pred_type == 'price':
                        # Color code price predictions
                        if value < 20:
                            st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🔴 (Low)")
                        elif value > 100:
                            st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🟡 (High)")
                        else:
                            st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🟢 (Normal)")
                    else:
                        st.write(f"{time.strftime('%H:%M')}: {value:.1f} MW")
                    if j >= 5:  # Show only first 6 predictions
                        break

# Data table section
if st.checkbox("Show Raw Data Table"):
    st.subheader("📋 Filtered Data")
    
    # Add download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"energy_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Display data with pagination
    page_size = st.select_slider("Rows per page:", [10, 25, 50, 100], value=25)
    
    # Pagination
    total_rows = len(filtered_df)
    total_pages = (total_rows - 1) // page_size + 1
    
    if total_pages > 1:
        page = st.number_input(
            f"Page (1 to {total_pages}):",
            min_value=1,
            max_value=total_pages,
            value=1
        )
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        display_df = filtered_df.iloc[start_idx:end_idx]
        
        st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    else:
        display_df = filtered_df
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Datetime": st.column_config.DatetimeColumn(
                "Date & Time",
                format="DD/MM/YYYY HH:mm"
            ),
            "Total_Wind": st.column_config.NumberColumn(
                "Wind Power (MW)",
                format="%.2f"
            ),
            "Total_Solar": st.column_config.NumberColumn(
                "Solar Power (MW)",
                format="%.2f"
            ),
            "Price_MWh": st.column_config.NumberColumn(
                "Price ($/MWh)",
                format="$%.2f"
            )
        }
    )

# Instructions
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    ### 🎯 Features:
    - **Time Period Selection**: Use the sidebar to select different time ranges
    - **Quick Presets**: Choose from common time periods or set custom ranges
    - **Multiple Chart Types**: Switch between time series, daily patterns, monthly patterns, and correlation analysis
    - **Interactive Charts**: Hover for details, zoom in/out, pan across time
    - **Data Export**: Download filtered data as CSV
    - **Responsive Design**: Charts adapt to your selected time period
    
    ### 🔮 **Prediction Modeling**:
    - **Machine Learning**: Uses Random Forest with time-based features
    - **24-48 Hour Forecasts**: Predict price and solar generation
    - **Dashed Lines**: Predictions shown in purple/red dashed lines
    - **Model Metrics**: MAE and RMSE displayed for model accuracy
    - **Feature Engineering**: Uses lag features, rolling statistics, and cyclical time encoding
    
    ### 📊 Chart Types:
    - **Time Series**: View data over time with multiple subplots
    - **Daily Patterns**: See average patterns throughout the day
    - **Monthly Patterns**: Compare seasonal variations
    - **Correlation Analysis**: Explore relationships between variables
    
    ### 💡 Tips:
    - Use the chart controls to zoom into specific periods
    - Toggle data series on/off using the sidebar checkboxes
    - Try different time periods to see patterns at various scales
    - Download data for further analysis in Excel or other tools
    """)