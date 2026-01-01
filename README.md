# AI Weather Forecasting with LSTMs

Copyright (c) 2026 Shrikara Kaudambady. All rights reserved.

## 1. Introduction

Weather forecasting is a quintessential time-series problem. Future weather conditions are deeply dependent on the sequence of atmospheric events that preceded them. While simpler models can provide basic forecasts, they often fail to capture the complex, long-term dependencies inherent in meteorological data.

This project implements a more sophisticated solution using a **Long Short-Term Memory (LSTM)** network, a specialized type of Recurrent Neural Network (RNN). LSTMs are designed to recognize patterns in sequential data, making them an excellent choice for tasks like weather prediction. This notebook demonstrates how to build, train, and evaluate an LSTM model to forecast future temperatures based on historical data.

## 2. The Solution Explained

The core of this solution is a deep learning model that processes sequences of past weather data to predict a future value.

### 2.1. Data Simulation

To create a self-contained example, the notebook begins by simulating a multi-year, hourly weather dataset. This synthetic data includes multiple variables to create a realistic, multi-variate time-series:
*   **Temperature:** Models both seasonal (winter/summer) and daily (day/night) variations.
*   **Humidity:** Modeled to have a partially inverse relationship with temperature.
*   **Pressure:** Simulated with a stable baseline and minor fluctuations.

### 2.2. Methodology: Sequence-to-Value Forecasting with LSTMs

The workflow is tailored specifically for training a neural network on time-series data.

1.  **Data Scaling:** All weather features (temperature, humidity, pressure) are normalized to a scale of [0, 1] using `scikit-learn`'s `MinMaxScaler`. This is a crucial step, as neural networks train most effectively on small, uniformly scaled values.

2.  **Data Sequencing:** This is the most critical preprocessing step for LSTMs. The time-series data is transformed into overlapping "windows". Each window consists of:
    *   **Input Sequence (X):** A sequence of data from the last `N` hours (e.g., 72 hours of temperature, humidity, and pressure).
    *   **Target Value (y):** The temperature at a future point in time (e.g., 12 hours after the input sequence ends).

3.  **LSTM Model Architecture:** A simple yet powerful LSTM model is built using the `TensorFlow` and `Keras` libraries:
    *   An `LSTM` layer acts as the "memory" of the network, processing the input sequence and learning its temporal patterns.
    *   A `Dropout` layer is included to prevent the model from overfitting to the training data.
    *   A `Dense` output layer produces the final, single-value temperature prediction.

4.  **Training and Evaluation:** The model is trained on the sequenced dataset. We then use it to make predictions on a held-out test set and "inverse transform" the predicted values back to the original Celsius scale to make them human-readable.

5.  **Visualization:** The notebook concludes by plotting the model's predictions directly against the actual temperatures from the test set, providing a clear visual assessment of the forecast's accuracy.

## 3. How to Use the Notebook

### 3.1. Prerequisites

You will need Python 3 and Jupyter Notebook/JupyterLab. This project also requires `TensorFlow`, a major deep learning library.

Install all necessary libraries with pip:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

### 3.2. Running the Notebook

1.  Clone this repository:
    ```bash
    git clone https://github.com/shrikarak/weather-forecasting-lstm-ai.git
    cd weather-forecasting-lstm-ai
    ```
2.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
3.  Open `weather_forecasting_lstm.ipynb` and run the cells sequentially.

## 4. Deployment and Customization

This notebook provides a robust template for time-series forecasting.

1.  **Using Your Own Data:**
    *   Replace the data simulation cell with code to load your own weather data (e.g., from a CSV file) into a pandas DataFrame.
    *   Ensure your DataFrame has a `datetime` index and columns for the relevant weather features.

2.  **Customizing the Forecast:**
    *   **Lookback Period:** Modify the `N_PAST` variable to change how many historical data points the model uses to make a prediction.
    *   **Forecast Horizon:** Change the `N_FUTURE` variable to predict further or closer into the future.
    *   **Model Architecture:** For more complex problems, you can experiment with the LSTM model by stacking more layers or increasing the number of units in each layer.
