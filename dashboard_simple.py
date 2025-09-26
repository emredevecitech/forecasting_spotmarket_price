import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Energy Data Dashboard", 
    page_icon="⚡", 
    layout="wide"
)

st.title("⚡ Energy Data Dashboard")
st.markdown("Interactive visualization of wind, solar, and pricing data with machine learning predictions")

@st.cache_data
def load_data():
    """Load the real energy data from CSV file"""
    try:
        # Load the CSV file with semicolon separator
        df = pd.read_csv("combined_energy_price_clean.csv", sep=';')
        
        # Convert Datetime column to datetime type
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Clean numeric columns
        numeric_columns = ['Total_Wind', 'Total_Solar', 'Price_MWh']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=['Datetime', 'Total_Wind', 'Total_Solar', 'Price_MWh'])
        
        # Data quality checks
        df = df[df['Total_Wind'] >= 0]
        df = df[df['Total_Solar'] >= 0]  
        df = df[df['Price_MWh'] >= 0]   
        df = df[df['Price_MWh'] <= 1000]
        df = df[df['Total_Wind'] <= 50000]
        df = df[df['Total_Solar'] <= 50000]
        
        # Extract time-based features
        if 'Hour' not in df.columns:
            df['Hour'] = df['Datetime'].dt.hour
        if 'DayOfWeek' not in df.columns:
            df['DayOfWeek'] = df['Datetime'].dt.dayofweek
        if 'Month' not in df.columns:
            df['Month'] = df['Datetime'].dt.month
        
        # Sort by datetime
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        st.error("❌ File 'combined_energy_price_clean.csv' not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data
with st.spinner("Loading your energy data..."):
    df = load_data()

if len(df) == 0:
    st.error("❌ No valid data remaining after cleaning.")
    st.stop()

# Simple prediction function
def create_simple_features(df, target_col):
    """Create simple features for prediction"""
    features_df = df.copy()
    
    # Time-based features
    features_df['hour'] = features_df['Datetime'].dt.hour
    features_df['day_of_week'] = features_df['Datetime'].dt.dayofweek
    features_df['month'] = features_df['Datetime'].dt.month
    
    # Cyclical encoding
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
    features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        if lag < len(features_df):
            features_df[f'{target_col}_lag_{lag}'] = features_df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12, 24]:
        if window < len(features_df):
            features_df[f'{target_col}_rolling_mean_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).mean()
            features_df[f'{target_col}_rolling_std_{window}'] = features_df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Cross-feature interactions
    if target_col == 'Price_MWh':
        features_df['total_generation'] = features_df['Total_Wind'] + features_df['Total_Solar']
        features_df['renewable_ratio'] = features_df['Total_Solar'] / (features_df['Total_Wind'] + features_df['Total_Solar'] + 1)
    
    return features_df

def train_simple_model(df, target_col, test_size=0.2):
    """Train a simple Random Forest model"""
    try:
        # Create features
        features_df = create_simple_features(df, target_col)
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            return None, None, None
        
        # Select feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Datetime', target_col] and not col.startswith('Total_')]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return model, scaler, {
            'mae': mae, 
            'rmse': rmse, 
            'r2': r2,
            'feature_cols': feature_cols
        }
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None

def make_simple_predictions(model, scaler, df, target_col, hours_ahead=24, feature_cols=None):
    """Make simple predictions"""
    try:
        # Get recent data
        recent_data = df.tail(48).copy()
        
        predictions = []
        current_data = recent_data.copy()
        
        # Generate future timestamps
        last_datetime = current_data['Datetime'].iloc[-1]
        future_datetimes = pd.date_range(
            start=last_datetime + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        for i, future_datetime in enumerate(future_datetimes):
            # Create future row
            future_row = current_data.iloc[-1:].copy()
            future_row['Datetime'] = future_datetime
            
            # Add to current data
            current_data = pd.concat([current_data, future_row], ignore_index=True)
            
            # Create features
            features_df = create_simple_features(current_data, target_col)
            
            if len(features_df) == 0:
                break
                
            # Get last row features
            latest_features = pd.DataFrame(index=[0])
            for col in feature_cols:
                if col in features_df.columns:
                    latest_features[col] = features_df[col].iloc[-1] if not pd.isna(features_df[col].iloc[-1]) else 0
                else:
                    latest_features[col] = 0
            
            latest_features = latest_features.fillna(0)
            
            # Scale and predict
            latest_features_scaled = scaler.transform(latest_features)
            pred = model.predict(latest_features_scaled)[0]
            
            # Apply constraints
            if target_col == 'Price_MWh':
                pred = max(10, min(500, pred))
            elif target_col in ['Total_Solar', 'Total_Wind']:
                pred = max(0, pred)
            
            predictions.append(pred)
            current_data.loc[current_data['Datetime'] == future_datetime, target_col] = pred
        
        return future_datetimes, predictions
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
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

# Calculate date ranges
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
    
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

# Filter data
filtered_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)].copy()

if filtered_df.empty:
    st.error("No data available for the selected time period.")
    st.stop()

# Display summary statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Data Points", f"{len(filtered_df):,}")

with col2:
    avg_wind = filtered_df['Total_Wind'].mean()
    st.metric("💨 Avg Wind", f"{avg_wind:.1f} MW")

with col3:
    avg_solar = filtered_df['Total_Solar'].mean()
    st.metric("☀️ Avg Solar", f"{avg_solar:.1f} MW")

with col4:
    avg_price = filtered_df['Price_MWh'].mean()
    st.metric("💰 Avg Price", f"${avg_price:.2f}/MWh")

# Prediction settings
st.sidebar.header("🔮 Prediction Settings")
enable_predictions = st.sidebar.checkbox("Enable Predictions", False)
if enable_predictions:
    prediction_hours = st.sidebar.selectbox("Prediction Horizon:", [24, 48], index=0)
    predict_price = st.sidebar.checkbox("Predict Price", True)
    predict_solar = st.sidebar.checkbox("Predict Solar", True)

# Prediction section
predictions_data = {}
if enable_predictions:
    if predict_price or predict_solar:
        st.subheader("🔮 Training Prediction Models")
        
        # Use recent data for training
        training_cutoff = filtered_df['Datetime'].max() - timedelta(days=365)
        training_data = df[df['Datetime'] >= training_cutoff].copy()
        
        if predict_price:
            with st.spinner("Training price prediction model..."):
                price_model, price_scaler, price_metrics = train_simple_model(training_data, 'Price_MWh')
                if price_model:
                    future_times, price_preds = make_simple_predictions(
                        price_model, price_scaler, training_data, 'Price_MWh', prediction_hours, price_metrics['feature_cols']
                    )
                    if future_times is not None:
                        predictions_data['price'] = {
                            'times': future_times,
                            'values': price_preds,
                            'metrics': price_metrics
                        }
                        st.success(f"✅ Price model trained - MAE: {price_metrics['mae']:.2f} $/MWh, R²: {price_metrics['r2']:.3f}")
        
        if predict_solar:
            with st.spinner("Training solar prediction model..."):
                solar_model, solar_scaler, solar_metrics = train_simple_model(training_data, 'Total_Solar')
                if solar_model:
                    future_times, solar_preds = make_simple_predictions(
                        solar_model, solar_scaler, training_data, 'Total_Solar', prediction_hours, solar_metrics['feature_cols']
                    )
                    if future_times is not None:
                        predictions_data['solar'] = {
                            'times': future_times,
                            'values': solar_preds,
                            'metrics': solar_metrics
                        }
                        st.success(f"✅ Solar model trained - MAE: {solar_metrics['mae']:.1f} MW, R²: {solar_metrics['r2']:.3f}")

# Chart options
st.sidebar.header("📈 Chart Options")
chart_type = st.sidebar.selectbox(
    "Chart Type:",
    ["Time Series", "Daily Patterns", "Monthly Patterns"]
)

show_wind = st.sidebar.checkbox("Show Wind Data", True)
show_solar = st.sidebar.checkbox("Show Solar Data", True)
show_price = st.sidebar.checkbox("Show Price Data", True)

# Create visualizations
if chart_type == "Time Series":
    # Create subplots
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
    
    # Add wind data
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
    
    # Add solar data
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
        
        # Add solar predictions if available
        if enable_predictions and 'solar' in predictions_data:
            fig.add_trace(
                go.Scatter(
                    x=predictions_data['solar']['times'], 
                    y=predictions_data['solar']['values'],
                    name='Solar (Predicted)', 
                    line=dict(color='#9467bd', width=3, dash='dash'),
                    hovertemplate='<b>Solar Prediction</b><br>Date: %{x}<br>Power: %{y:.1f} MW<extra></extra>'
                ), 
                row=current_row, col=1
            )
        
        fig.update_yaxes(title_text="Solar Power (MW)", row=current_row, col=1)
        current_row += 1
    
    # Add price data
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
        
        # Add price predictions if available
        if enable_predictions and 'price' in predictions_data:
            fig.add_trace(
                go.Scatter(
                    x=predictions_data['price']['times'], 
                    y=predictions_data['price']['values'],
                    name='Price (Predicted)', 
                    line=dict(color='#d62728', width=3, dash='dash'),
                    hovertemplate='<b>Price Prediction</b><br>Date: %{x}<br>Price: $%{y:.2f}/MWh<extra></extra>'
                ), 
                row=current_row, col=1
            )
        
        fig.update_yaxes(title_text="Price ($/MWh)", row=current_row, col=1)
    
    # Update layout
    chart_title = f"Energy Data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    if enable_predictions and predictions_data:
        chart_title += f" (+ {prediction_hours}h Predictions)"
    
    fig.update_layout(
        height=200 * num_subplots + 100,
        title_text=chart_title,
        title_x=0.5,
        hovermode='x unified',
        showlegend=True
    )

elif chart_type == "Daily Patterns":
    # Group by hour
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

else:  # Monthly Patterns
    # Group by month
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

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Show prediction metrics if available
if enable_predictions and predictions_data:
    st.subheader("📊 Prediction Model Performance")
    
    if 'price' in predictions_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Price Model MAE", f"${predictions_data['price']['metrics']['mae']:.2f}/MWh")
        with col2:
            st.metric("💰 Price Model RMSE", f"${predictions_data['price']['metrics']['rmse']:.2f}/MWh")
        with col3:
            st.metric("💰 Price Model R²", f"{predictions_data['price']['metrics']['r2']:.3f}")
    
    if 'solar' in predictions_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("☀️ Solar Model MAE", f"{predictions_data['solar']['metrics']['mae']:.1f} MW")
        with col2:
            st.metric("☀️ Solar Model RMSE", f"{predictions_data['solar']['metrics']['rmse']:.1f} MW")
        with col3:
            st.metric("☀️ Solar Model R²", f"{predictions_data['solar']['metrics']['r2']:.3f}")
    
    # Show next few prediction values
    st.subheader("🔮 Next Few Hours Predictions")
    
    if 'price' in predictions_data:
        st.write("**Price Predictions:**")
        for j, (time, value) in enumerate(zip(predictions_data['price']['times'][:6], predictions_data['price']['values'][:6])):
            if value < 20:
                st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🔴 (Low)")
            elif value > 100:
                st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🟡 (High)")
            else:
                st.write(f"{time.strftime('%H:%M')}: ${value:.2f}/MWh 🟢 (Normal)")
            if j >= 5:
                break
    
    if 'solar' in predictions_data:
        st.write("**Solar Predictions:**")
        for j, (time, value) in enumerate(zip(predictions_data['solar']['times'][:6], predictions_data['solar']['values'][:6])):
            st.write(f"{time.strftime('%H:%M')}: {value:.1f} MW")
            if j >= 5:
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
    - **Multiple Chart Types**: Switch between time series, daily patterns, and monthly patterns
    - **Interactive Charts**: Hover for details, zoom in/out, pan across time
    - **Data Export**: Download filtered data as CSV
    - **Machine Learning**: Random Forest predictions for price and solar generation
    
    ### 🔮 **Prediction Modeling**:
    - **Random Forest**: Uses time-based features and lag features
    - **24-48 Hour Forecasts**: Predict price and solar generation
    - **Dashed Lines**: Predictions shown in purple/red dashed lines
    - **Model Metrics**: MAE, RMSE, and R² displayed for model accuracy
    
    ### 📊 Chart Types:
    - **Time Series**: View data over time with multiple subplots
    - **Daily Patterns**: See average patterns throughout the day
    - **Monthly Patterns**: Compare seasonal variations
    
    ### 💡 Tips:
    - Use the chart controls to zoom into specific periods
    - Toggle data series on/off using the sidebar checkboxes
    - Try different time periods to see patterns at various scales
    - Download data for further analysis in Excel or other tools
    """)

