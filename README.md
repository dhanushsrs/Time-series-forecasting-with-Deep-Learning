Advanced Time Series Forecasting with Attention (PyTorch)

This project implements multivariate time series forecasting using LSTM-based Seq2Seq models, both with and without Bahdanau Attention, using PyTorch.
The goal is to compare a baseline LSTM model against an attention-enhanced model and analyze performance and interpretability.

Project Overview

Synthetic multivariate time series data generation

Data scaling and sliding window sequence creation

Baseline Seq2Seq LSTM model

Seq2Seq LSTM with Bahdanau Attention

Model evaluation using RMSE, MAE, and MAPE

Visualization of attention weights over time steps

Dataset Description

The dataset is synthetically generated and consists of:

Three input features: f1, f2, f3

One target variable: y

Combination of trend, seasonality, and noise

Total data points: 1500

Sequence length: 30

Model Details
Baseline Model

LSTM encoder

Uses the final hidden state for prediction

Fully connected layer for output

Attention Model

LSTM encoder

Bahdanau (additive) attention mechanism

Learns to focus on important past time steps

Context vector used for final prediction

Evaluation Metrics

The models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

These metrics help compare prediction accuracy between baseline and attention models.

Requirements

Install the required Python libraries:

pip install numpy pandas torch scikit-learn matplotlib

How to Run

Clone the repository:

git clone https://github.com/your-username/attention-time-series-forecasting.git


Navigate to the project directory:

cd attention-time-series-forecasting


Run the program:

python attention_time_series.py


The script trains both models, evaluates them, and displays attention weights.

Key Learning Outcomes

Understanding Seq2Seq models for time series

Implementing attention mechanisms in PyTorch

Comparing traditional LSTM vs attention-based models

Interpreting attention weights for explainability

Applications

Financial time series forecasting

Weather prediction

Energy consumption forecasting

Sensor and IoT data analysis

Any multivariate sequential data

Future Enhancements

Multi-step forecasting

Real-world datasets

Transformer-based models

Hyperparameter optimization

Validation loss visualization
