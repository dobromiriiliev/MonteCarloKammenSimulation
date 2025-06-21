# Kalman Filter & Monte Carlo Simulation for NASDAQ Tracking

 **Market Modeling Using Kalman Filtering and Monte Carlo Forecasting**

This project implements a Kalman filter to track and predict NASDAQ closing prices for the year 2008. It combines probabilistic state estimation with stochastic simulation to analyze and forecast financial trends using real-world economic data.

## Project Overview

The model incorporates both a Kalman filter and a Monte Carlo simulation to enhance the forecasting of market behavior:

- **Kalman Filter**: Estimates the dynamic state of the system by predicting and updating the value of the beta parameter using sequential NASDAQ closing prices. This recursive filtering method is ideal for noisy time-series data.
  
- **Autoregressive (AR) Process**: Captures the temporal dependencies in the NASDAQ dataset, modeling how past values influence future prices.

- **Monte Carlo Simulation**: Simulates random variations in a leading economic indicator (LEI), accounting for uncertainty and variability in economic trends. This component adds robustness and a probabilistic forecast to the model.

## Visualization

The analysis concludes with:
- A plot comparing actual vs. Kalman-predicted NASDAQ closing values.
- A separate visualization of the AR process over time.
- A time series plot of the Monte Carlo simulation applied to the LEI.

These visualizations provide intuitive insights into how well the model tracks market behavior and how economic indicators influence forecasts.

---

Let me know if you’d like installation instructions, example output plots, or additional enhancements like interactive dashboards or CSV export features.
