# ⚡ Energy Data Dashboard

An interactive Streamlit dashboard for analyzing wind, solar, and energy pricing data with advanced machine learning predictions using multiple models.

## 🚀 Features

### 📊 Data Visualization
- **Interactive Time Series Charts** with Plotly
- **Multiple Chart Types**: Time series, daily patterns, monthly patterns, correlation analysis
- **Real-time Data Filtering** with customizable time periods
- **Responsive Design** that adapts to your data

### 🤖 Machine Learning Predictions
- **Multiple Model Support**:
  - **Random Forest** 🟢 - Traditional machine learning
  - **LSTM Neural Network** 🟠 - Deep learning for time series
  - **ARIMA** 🔵 - Statistical time series modeling
  - **Prophet** 🟣 - Facebook's forecasting tool

- **Advanced Features**:
  - **Model Comparison** with side-by-side performance metrics
  - **Color-coded Predictions** for easy visualization
  - **Feature Importance Analysis**
  - **Automatic Best Model Selection**

### 📈 Prediction Capabilities
- **Price Forecasting** with realistic constraints
- **Solar Generation Prediction** with daily patterns
- **Wind Generation Prediction** with weather patterns
- **24-48 Hour Forecasts** with confidence intervals

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.11
- pip package manager

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/energy-dashboard.git
   cd energy-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

## 📁 Project Structure

```
energy-dashboard/
├── dashboard.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
├── combined_energy_price_clean.csv  # Main dataset
└── data/                    # Additional data files
    ├── wind_data.csv
    ├── solar_data.csv
    └── price_data.csv
```

## 🎯 Usage

### 1. **Data Loading**
- The dashboard automatically loads data from `combined_energy_price_clean.csv`
- Supports CSV files with columns: `Datetime`, `Total_Wind`, `Total_Solar`, `Price_MWh`

### 2. **Time Period Selection**
- **Quick Presets**: Last 7/30/90 days, Last Year, Full Dataset
- **Custom Range**: Select specific start and end dates
- **Year-specific**: 2023/2024 data filters

### 3. **Model Training**
- **Enable Predictions**: Toggle prediction features
- **Model Selection**: Choose which models to compare
- **Training Progress**: Real-time training status

### 4. **Visualization**
- **Chart Types**: Switch between different visualization modes
- **Data Toggles**: Show/hide wind, solar, and price data
- **Interactive Features**: Zoom, pan, hover for details

## 🔧 Model Details

### Random Forest
- **Best for**: General-purpose prediction with feature engineering
- **Features**: Time-based features, lag features, rolling statistics
- **Strengths**: Handles non-linear relationships, feature importance

### LSTM Neural Network
- **Best for**: Complex time series patterns and sequences
- **Architecture**: 2-layer LSTM with dropout regularization
- **Strengths**: Captures long-term dependencies, handles seasonality

### ARIMA
- **Best for**: Statistical time series with trends and seasonality
- **Auto-selection**: Automatically finds optimal (p,d,q) parameters
- **Strengths**: Interpretable, handles stationarity

### Prophet
- **Best for**: Business forecasting with holidays and seasonality
- **Features**: Daily, weekly, yearly seasonality detection
- **Strengths**: Robust to missing data, handles holidays

## 📊 Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors
- **R² (R-squared)**: Proportion of variance explained
- **Feature Importance**: Which features matter most

## 🚀 Deployment

### Streamlit Cloud
1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Connect your GitHub repository
4. **Configure**: Set main file to `dashboard.py`

### Local Deployment
```bash
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

## 📈 Data Requirements

Your CSV file should have these columns:
- `Datetime`: Timestamp in YYYY-MM-DD HH:MM:SS format
- `Total_Wind`: Wind generation in MW
- `Total_Solar`: Solar generation in MW  
- `Price_MWh`: Energy price in $/MWh

## 🔍 Troubleshooting

### TensorFlow Issues
If you see TensorFlow errors:
```bash
pip uninstall tensorflow
pip install tensorflow==2.16.1
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data Format Issues
- Ensure datetime column is properly formatted
- Check for missing values in critical columns
- Verify numeric columns don't have text values

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Scikit-learn** for machine learning tools
- **TensorFlow** for deep learning capabilities
- **Statsmodels** for statistical modeling
- **Prophet** for time series forecasting

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the error messages in the dashboard
3. Ensure all dependencies are installed
4. Verify your data format matches requirements

---

**Made with ❤️ for Energy Data Analysis**
