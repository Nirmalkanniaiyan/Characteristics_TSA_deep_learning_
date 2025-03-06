from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return [mae, mape, rmse, r2]

def prepare_data(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
        """Prepare data for modeling"""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df = df.sort_index()
        return df

import pandas as pd

def format_results(all_results):
    formatted_results = {}

    for result in all_results:
        file_name = result[0]  # First column is the file name
        features = ["Stationarity", "Seasonality", "Trend", "Volatility"]
        feature_values = result[1:5]  # Next four columns are feature values
        
        metrics = ["MAE", "MAPE", "RMSE", "R2"]
        lstm_values = result[5:9]   # LSTM metrics
        gru_values = result[9:13]   # GRU metrics
        cnn_values = result[13:17]  # CNN metrics
        
        # Create a DataFrame in the required format
        df = pd.DataFrame({
            "Feature": features,
            "Feature Value": list(feature_values),
            "Metrics": metrics,
            "LSTM": lstm_values,
            "CNN": cnn_values,
            "GRU": gru_values
        })
        
        formatted_results[file_name] = df

    return formatted_results

# Example Usage
# formatted_results = format_results(all_results)
# Now `formatted_results` is a dictionary where keys are filenames, and values are formatted DataFrames.
