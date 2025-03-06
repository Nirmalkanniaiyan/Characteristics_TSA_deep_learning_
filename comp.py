import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
import logging
from pymannkendall import original_test
import itertools
import plotly.graph_objects as go

class TimeSeriesAnalyzer:
    """Class to analyze time series characteristics"""
    
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target = data[target_column]
        self.characteristics = {}
        
    def check_stationarity(self) -> Dict:
        """
        Check stationarity using both ADF and KPSS tests
        ADF H0: Series has unit root (non-stationary)
        KPSS H0: Series is trend stationary
        """
        # ADF Test
        adf_result = adfuller(self.target)
        
        # KPSS Test
        kpss_result = kpss(self.target)
        
        is_stationary = (adf_result[1] < 0.05) and (kpss_result[1] >= 0.05)
        
        return {
            'is_stationary': is_stationary,
            'adf_pvalue': adf_result[1],
            'kpss_pvalue': kpss_result[1]
        }
    
    def check_seasonality(self) -> Dict:
        """Check for presence of seasonality using ACF."""
    
        # Ensure the target index is a DateTimeIndex
        if not pd.api.types.is_datetime64_any_dtype(self.target.index):
            return {
                'is_seasonal': False,
                'seasonal_strength': None,
                'suggested_period': None,
                'message': "The index of the target series is not a DateTimeIndex."
            }

        # Calculate ACF values
        lag_acf = acf(self.target, nlags=40)  # Adjust the number of lags based on your needs

        # Define a threshold for significance (e.g., 0.2)
        threshold = 0.2

        # Check for significant lags
        significant_lags = np.where(lag_acf > threshold)[0]
        
        # Determine if seasonality exists based on significant lags
        is_seasonal = len(significant_lags) > 0

        # If seasonality exists, find the period (assuming periodicity occurs)
        suggested_period = None
        if is_seasonal:
            # Assuming the first significant lag indicates the seasonal period
            suggested_period = significant_lags[0]

        return {
            'is_seasonal': is_seasonal,
            'significant_lags': significant_lags.tolist(),  # Convert to list for better readability
            'seasonal_strength': None,  # You can calculate this if needed
            'suggested_period': suggested_period,
            'message': "Seasonality check completed using ACF."
        }
    def check_trend(self) -> Dict:
        
        trend_result = original_test(self.target)  # Use Mann-Kendall on raw data
        
        has_trend = trend_result.p < 0.05
        trend_strength = trend_result.z  # Z-score indicates trend strength

        return {
            'has_trend': has_trend,
            'trend_strength': trend_strength,
            'slope': trend_result.slope,
            'p_value': trend_result.p
        }
    
    def check_volatility(self) -> Dict:
        """Check for volatility in the series"""
        rolling_std = self.target.rolling(window=min(len(self.target)//4, 30)).std()
        volatility_ratio = rolling_std.max() / rolling_std.min()
        
        high_volatility = volatility_ratio > 2.0
        
        return {
            'high_volatility': high_volatility,
            'volatility_ratio': volatility_ratio
        }
    
    def analyze(self) -> Dict:
        """Run all analysis"""
        self.characteristics['stationarity'] = self.check_stationarity()
        self.characteristics['seasonality'] = self.check_seasonality()
        self.characteristics['trend'] = self.check_trend()
        self.characteristics['volatility'] = self.check_volatility()
        
        return self.characteristics

class ModelSelector:
    """Class to select appropriate model based on time series characteristics"""
    
    @staticmethod
    def select_model(characteristics: Dict) -> str:
        """
        Select model based on time series characteristics.
        Returns: model_name, reason.
        """
        is_stationary = characteristics['stationarity']['is_stationary']
        # p_value = characteristics['stationarity']['p_value']  # Use p-value for better thresholding
        is_seasonal = characteristics['seasonality']['is_seasonal']
        has_trend = characteristics['trend']['has_trend']
        trend_strength = characteristics['trend']['trend_strength']
        high_volatility = characteristics['volatility']['high_volatility']
        
        if is_seasonal and has_trend:
            if high_volatility:
                print("1") 
                return "prophet", "Selected due to presence of seasonality, trend, and high volatility"
            else:
                print("2") 
                return "sarima", "Selected due to presence of seasonality and trend with moderate volatility"
        
        elif has_trend and not is_seasonal:
            if high_volatility:
                print("3") 
                return "prophet", "Selected due to presence of trend and high volatility"
            else:
                print("4") 
                return "holt_winters", "Selected due to presence of trend with moderate volatility"
        
        elif is_stationary:
            print("6") 
            return "arima", "Selected due to stationarity"
        
        elif is_seasonal and not has_trend:
            print("5") 
            return "sarima", "Selected due to presence of seasonality without significant trend"
        
            # else:
            #     return "holt_winters", "Selected due to weak stationarity"
        
        else:
            print("7") 
            # Consider adding additional checks for autocorrelation here
            return "prophet", "Selected as default due to no strong characteristics"

class AdvancedTimeSeriesForecaster:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.characteristics = None
        self.selection_reason = None
        self.train = None
        self.test = None
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
        """Prepare data for modeling"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df = df.sort_index()
        return df
    
    def train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        split_idx = int(len(df) * (1 - test_size))
        self.train = df[:split_idx]
        self.test = df[split_idx:]
        return self.train, self.test
    
    def fit_sarima(self, train: pd.DataFrame, target_column: str):
        """Fit SARIMA model with automated parameter selection."""
    
        # Retrieve suggested seasonal period or set a default if invalid
        suggested_period = self.characteristics['seasonality'].get('suggested_period', 12)  # default to 12
        if suggested_period is None or suggested_period <= 1:
            suggested_period = 12  # Default to a 12-period seasonality if none is valid

        # Define the p, d, q parameters to take values between 0 and 2
        p = d = q = range(0, 2)  # Only test 0 and 1 for each parameter
        seasonal_order = range(0, 2)  # Only 0 and 1 for seasonal parameters
        
        # Create list of all combinations of p, d, q
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(seasonal_order, seasonal_order, seasonal_order, [suggested_period]))

        best_aic = float("inf")
        best_params = None
        best_seasonal_params = None

        for param in pdq:
            for seasonal_param in seasonal_pdq:
                try:
                    model = SARIMAX(train[target_column], order=param, seasonal_order=seasonal_param)
                    results = model.fit(disp=False)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = param
                        best_seasonal_params = seasonal_param
                except Exception as e:
                    continue  # Skip problematic parameter combinations

        if best_params is None or best_seasonal_params is None:
            raise ValueError("No suitable SARIMA model found.")

        model = SARIMAX(
        train[target_column],
        order=param,
        seasonal_order=seasonal_param
        )
        results = model.fit(disp=False, maxiter=50)  # Limit max iterations
        return results
        
    def fit_prophet(self, train: pd.DataFrame, target_column: str):
        """Fit Prophet model"""
        prophet_data = pd.DataFrame({
            'ds': train.index,
            'y': train[target_column]
        })
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_data)
        return model
    
    def fit_holt_winters(self, train: pd.DataFrame, target_column: str):
        """Fit Holt-Winters model"""
        model = ExponentialSmoothing(
            train[target_column],
            trend='add',
            seasonal=None,
            damped=True
        )
        return model.fit()
    
    def check_stationarity(self, series: pd.Series) -> int:
        """Check if the series is stationary and determine the order of differencing (d)."""
        result = adfuller(series)
        if result[1] < 0.05:
            logging.info("Data is stationary.")
            return 0
        else:
            logging.info("Data is non-stationary, applying differencing.")
            return 1 

    def fit_arima(self, train: pd.DataFrame, target_column: str, p_range=range(0, 4), q_range=range(0, 4)):
        # Step 1: Check stationarity and determine d
        d = self.check_stationarity(train[target_column])
        
        # Step 2: Determine optimal (p, d, q) values
        best_aic = np.inf
        best_order = None
        best_model = None

        for p in p_range:
            for q in q_range:
                try:
                    # Fit the ARIMA model
                    model = SARIMAX(train[target_column], order=(p, d, q))
                    model_fit = model.fit(disp=False)
                    
                    # Compare AIC values
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                        logging.info(f"New best model found: order={best_order} AIC={best_aic}")
                        
                except Exception as e:
                    logging.warning(f"Model with order {(p, d, q)} failed to fit: {e}")
                    continue  # Skip any models that fail to fit

        if best_model:
            logging.info(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
        else:
            logging.error("No suitable ARIMA model found.")
        
        return best_model  # Return the fitted model with the best parameters
    
    def get_forecast(self, model, test: pd.DataFrame, target_column: str):
        """Get forecast values based on model type"""
        if self.model_name == 'prophet':
            future_dates = pd.DataFrame({'ds': test.index})
            forecast = model.predict(future_dates)['yhat']
        elif self.model_name in ['sarima', 'arima']:
            forecast = model.forecast(len(test))
        else:  # holt_winters
            forecast = model.forecast(len(test))
        return forecast
    
    def fit(self, df: pd.DataFrame, date_column: str, target_column: str):
        """Analyze data, select and fit appropriate model"""
        # Prepare data
        prepared_data = self.prepare_data(df, date_column, target_column)
        self.train, self.test = self.train_test_split(prepared_data)
        
        # Analyze time series characteristics
        analyzer = TimeSeriesAnalyzer(prepared_data, target_column)
        self.characteristics = analyzer.analyze()
        
        # Select appropriate model
        self.model_name, self.selection_reason = ModelSelector.select_model(self.characteristics)
        
        # Fit selected model
        if self.model_name == 'sarima':
            self.model = self.fit_sarima(self.train, target_column)
        elif self.model_name == 'prophet':
            self.model = self.fit_prophet(self.train, target_column)
        elif self.model_name == 'holt_winters':
            self.model = self.fit_holt_winters(self.train, target_column)
        elif self.model_name == 'arima':
            self.model = self.fit_arima(self.train, target_column)
            
        # Get forecast and calculate metrics
        forecast = self.get_forecast(self.model, self.test, target_column)
        mae = mean_absolute_error(self.test[target_column], forecast)
        rmse = np.sqrt(mean_squared_error(self.test[target_column], forecast))
        
        self.metrics = {
            'mae': mae,
            'rmse': rmse,
            'forecast': forecast
        }
        
    def plot_results(self, target_column: str):
        """Plot actual vs forecast values and display analysis results using Plotly."""
        if self.test is None or self.model is None:
            raise ValueError("Must call fit() before plotting results")
        
        # Ensure metrics['forecast'] matches the length of the test set
        if len(self.metrics['forecast']) != len(self.test):
            raise ValueError(f"Length of forecast ({len(self.metrics['forecast'])}) does not match length of test set ({len(self.test)})")
        
        # Combine actual and forecasted data into a DataFrame
        combined_data = pd.DataFrame({
            'actual': pd.concat([self.train[target_column], self.test[target_column]]),
            'forecast': pd.Series(np.concatenate([np.full(len(self.train), np.nan), self.metrics['forecast']]),
                                index=self.train.index.append(self.test.index))
        })
        
        # Plot Actual vs Forecast using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['actual'], 
                                mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['forecast'], 
                                mode='lines', name=f'Forecast ({self.model_name})', line=dict(color='red', dash='dash')))
        fig.update_layout(title="Actual vs Forecast Values", xaxis_title="Date", yaxis_title="Value")
        
        # Add Model Characteristics and Performance Metrics as annotations
        characteristics_text = (
            f"Model Selected: {self.model_name.upper()}<br>"
            f"Selection Reason: {self.selection_reason}<br><br>"
            f"Time Series Characteristics:<br>"
            f"- Stationary: {self.characteristics['stationarity']['is_stationary']}<br>"
            f"- Seasonal: {self.characteristics['seasonality']['is_seasonal']}<br>"
            f"  (Strength: {self.characteristics['seasonality']['seasonal_strength'] or 0:.2f})<br>"
            f"- Trend: {self.characteristics['trend']['has_trend']}<br>"
            f"  (Strength: {self.characteristics['trend']['trend_strength'] or 0:.2f})<br>"
            f"- High Volatility: {self.characteristics['volatility']['high_volatility']}<br>"
            f"  (Ratio: {self.characteristics['volatility']['volatility_ratio'] or 0:.2f})<br><br>"
            f"Model Performance:<br>"
            f"- MAE: {self.metrics['mae'] or 0:.2f}<br>"
            f"- RMSE: {self.metrics['rmse'] or 0:.2f}"
        )

        # st.write(characteristics_text)
        
        # fig.add_annotation(
        #     xref="paper", yref="paper", x=0.5, y=-0.2,
        #     text=characteristics_text, 
        #     showarrow=False, align="left", 
        #     font=dict(family="Courier New, monospace", size=12, color="black"),
        #     bordercolor="black", borderwidth=1, borderpad=10, bgcolor="white",
        # )
        
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=100))
        fig.show()

        return [fig, characteristics_text]