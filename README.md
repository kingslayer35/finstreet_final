#  Finstreet: ML-Driven Financial Backtesting & Analysis System

<div align="center">

**An advanced Python-based framework for quantitative financial analysis, strategy backtesting, and machine learning-driven trading signal generation.**

</div>

# Project Overview

This project implements a production-style algorithmic trading pipeline that converts raw historical market data into actionable buy/sell signals, evaluates strategy robustness through realistic backtesting, and emphasizes risk management and reproducibility.

The system is built with a strong focus on:

Methodological rigor
Realistic trading constraints
Explainable ML-driven decision making

## Quick Start

Follow these steps to get Finstreet up and running on your local machine.

### Prerequisites

-   **Python 3.9+** is required.
    -   You can download it from [python.org](https://www.python.org/downloads/).
    -   It's recommended to use a virtual environment.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/kingslayer35/finstreet_final.git
    cd finstreet_final
    ```

2.  **Create and activate a virtual environment (recommended)**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Results

### Cumulative Returns
<img src="output/figures/cumulative_returns.png" width="800">

### Trade Performance
<img src="output/figures/performance_donut.png" width="800">

### Profit and Loss
<img src="output/figures/pnl_histogram.png" width="800">

### Trade Execution
<img src="output/figures/trade_execution.png" width="800">

### Trade Performance
<img src="output/figures/trade_scatter.png" width="800">

## Forward Predictions (Jan 1-8, 2026)

| Date       | Signal | Direction | Confidence | Position Size | Stop Loss | Take Profit |
|------------|--------|-----------|------------|---------------|-----------|-------------|
| 2026-01-01 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |
| 2026-01-02 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |
| 2026-01-03 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |
| 2026-01-06 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |
| 2026-01-07 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |
| 2026-01-08 | HOLD   | UP        | 50.69%     | 0.00%         | 0.0       | 0.0         |


# Project Structure

```
finstreet_final/
├── backtester.py       # Core logic for running backtests and evaluating strategies.
├── charts.py           # Functions for generating various financial charts and visualizations.
├── config.py           # Centralized configuration settings for the entire project.
├── data/               # Directory for storing raw or processed historical market data (e.g., CSV files).
├── data_loader.py      # Handles fetching and loading financial data into a usable format (e.g., from yfinance).
├── indicators.py       # Implementations of various technical analysis indicators.
├── labels.py           # Logic for creating target labels for machine learning models.
├── main.py             # The primary entry point for executing the financial analysis and backtesting system.
├── ml_ensemble.py      # Module for training, evaluating, and using machine learning ensemble models.
├── models/             # Directory for saving and loading trained machine learning models.
├── output/             # Stores generated backtest reports, charts, and other analytical outputs.
├── requirements.txt    # Lists all Python dependencies required for the project.
├── results.py          # Functions for analyzing and summarizing backtesting results.
└── signals.py          # Logic for generating trading signals based on indicators or ML predictions.
```
