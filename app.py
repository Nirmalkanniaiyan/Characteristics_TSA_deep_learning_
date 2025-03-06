import streamlit as st
import pandas as pd
from models.single_lstm import single_LSTM
from models.single_gru import single_GRU
from models.single_cnn import single_CNN
from comp import TimeSeriesAnalyzer
from utility import prepare_data,format_results

st.title('Automated Time Series Forecasting')
st.write("## We are experimenting with 3 models: LSTM, GRU, and CNN")

uploaded_files = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_results = []

    for uploaded_file in uploaded_files:
        result = [uploaded_file.name]
        df = pd.read_csv(uploaded_file)

        st.write("# Original Data")
        st.write(df)

        st.write("# Finding characteristics of the data")

        # Prepare data
        date_column = df.columns[0]
        target_column = df.columns[1]
        prepared_data = prepare_data(df, date_column, target_column)

        # Analyze time series characteristics
        analyzer = TimeSeriesAnalyzer(prepared_data, target_column)
        characteristics = analyzer.analyze()

        char_results = {
            "Stationarity": characteristics["stationarity"]["is_stationary"],
            "Seasonality": characteristics["seasonality"]["is_seasonal"],
            "Trend": characteristics["trend"]["has_trend"],
            "Volatility": characteristics["volatility"]["high_volatility"]
        }

        for values in char_results.values():
            result.append(values)

        st.write("# Forecasting data")
        data = df[df.columns[1]]

        st.write("## LSTM Model")
        lstm_metrics = single_LSTM(data)
        result.extend(lstm_metrics)

        st.write("## GRU Model")
        gru_metrics = single_GRU(data)
        result.extend(gru_metrics)

        st.write("## CNN Model")
        cnn_metrics = single_CNN(data)
        result.extend(cnn_metrics)

        # Assign points to determine the best model
        models = {"LSTM": lstm_metrics, "GRU": gru_metrics, "CNN": cnn_metrics}
        points = {"LSTM": 0, "GRU": 0, "CNN": 0}

        # Best model per metric
        for metric_idx in range(4):  # 4 metrics (MAE, MAPE, RMSE, R2)
            metric_values = {model: metrics[metric_idx] for model, metrics in models.items()}

            if metric_idx == 3:  # R² (higher is better)
                best_model = max(metric_values, key=metric_values.get)
            else:  # MAE, MAPE, RMSE (lower is better)
                best_model = min(metric_values, key=metric_values.get)

            points[best_model] += 1

        # Select model with the highest points
        best_model = max(points, key=points.get)
        result.append(best_model)

        all_results.append(result)

    # Create a DataFrame with a "Best Model" column
    columns = ["File", "Stationarity", "Seasonality", "Trend", "Volatility",
               "LSTM MAE", "LSTM MAPE", "LSTM RMSE", "LSTM R2",
               "GRU MAE", "GRU MAPE", "GRU RMSE", "GRU R2",
               "CNN MAE", "CNN MAPE", "CNN RMSE", "CNN R2", "Best Model"]

    df_all_results = pd.DataFrame(all_results, columns=columns)

    st.write("# Summary of all results")
    st.dataframe(df_all_results.style.set_table_styles(
        [{"selector": "th", "props": [("font-size", "16px"), ("text-align", "center")]}]
    ))

    st.write("# Download the results")
    csv = df_all_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )

    st.write("# Formatted Summary of all results")
    format_results = format_results(all_results)
    for key in format_results.keys():
        st.write(key)
        st.dataframe(format_results[key].style.set_table_styles(
            [{"selector": "th", "props": [("font-size", "16px"), ("text-align", "center")]}]
        ))
