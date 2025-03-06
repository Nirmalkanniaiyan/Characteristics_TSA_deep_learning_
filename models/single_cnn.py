import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import time

from sklearn.preprocessing import MinMaxScaler
from utility import compute_metrics

# Suppress the deprecation warning for plt.show() usage
st.set_option('deprecation.showPyplotGlobalUse', False)

# def df_to_X_y(df, window_size=5):
#   df_as_np = df.to_numpy()
#   X = []
#   y = []
#   for i in range(len(df_as_np)-window_size):
#     row = [[a] for a in df_as_np[i:i+window_size]]
#     X.append(row)
#     label = df_as_np[i+window_size]
#     y.append(label)
#   return np.array(X), np.array(y)

def df_to_X_y(df, window_size=5):

  pandas_df = pd.DataFrame(df)
  column_name = pandas_df.columns[0]
  df_reshape = pandas_df[column_name].values.reshape(-1, 1)  # Reshape to 2D array

  scaler = MinMaxScaler()
  df_as_np_scaled = scaler.fit_transform(df_reshape)

  X = []
  y = []
  for i in range(len(df_as_np_scaled)-window_size):
      row = [[a] for a in df_as_np_scaled[i:i+window_size]]
      X.append(row)
      label = df_as_np_scaled[i+window_size]
      y.append(label)
  return np.array(X), np.array(y)

def single_CNN(temp):
    model = Sequential([
        InputLayer((5, 1)),
        Conv1D(64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    cp = ModelCheckpoint('model.keras', save_best_only=True)
    history = History()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

    WINDOW_SIZE = 5
    X, y = df_to_X_y(temp, WINDOW_SIZE)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    epochs = 10
    progress_bar = st.progress(0)
    epoch_text = st.empty()
    loss_text = st.empty()
    val_loss_text = st.empty()

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, callbacks=[cp], verbose=0)
        time.sleep(1)  # Simulate training time

        # Update progress bar, epoch info, loss and validation loss
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        epoch_text.text(f"Epoch: {epoch + 1}")
        loss_text.text(f"Loss: {history.history['loss'][-1]:.4f}")
        val_loss_text.text(f"Val Loss: {history.history['val_loss'][-1]:.4f}")

    model = load_model('model.keras')

    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Test Predictions': test_predictions.flatten(), 'Actuals': y_test.flatten()})
    st.write(test_results)

    # Calculate and display additional metrics
    # mae = mean_absolute_error(y_test, test_predictions)
    # r2 = r2_score(y_test, test_predictions)

    # st.write(f"Mean Absolute Error: {mae:.4f}")
    # st.write(f"R² Score: {r2:.4f}")

    plt.plot(test_results['Test Predictions'][:50], label='Test Predictions')
    plt.plot(test_results['Actuals'][:50], label='Actuals')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Comparison of Test Predictions and Actuals')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot()

    # Return the final test results DataFrame for potential further analysis or display
    result_metrics = compute_metrics(y_test, test_predictions)
    return result_metrics