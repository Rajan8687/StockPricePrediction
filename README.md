
## Notebook: `stock.ipynb`

### Purpose

The `stock.ipynb` notebook contains the complete workflow for:
1.  Fetching historical stock data for a specific ticker (AAPL in this example).
2.  Preprocessing the data for time series forecasting.
3.  Building, training, and evaluating an LSTM model.
4.  Visualizing the model's performance.

### Libraries Used

The primary libraries used in the notebook are:
-   `numpy`
-   `pandas`
-   `matplotlib.pyplot`
-   `seaborn`
-   `yfinance`
-   `scikit-learn` (specifically `MinMaxScaler`)
-   `tensorflow.keras` (for `Sequential` model, `LSTM`, and `Dense` layers)

### Key Steps

1.  **Import Libraries:** Essential packages are imported.
2.  **Data Acquisition:**
    -   Historical stock data for 'AAPL' (Apple Inc.) is downloaded using `yfinance` for the period 2015-01-01 to 2023-12-31.
    -   Only the 'Close' price is selected for prediction.
3.  **Data Preprocessing:**
    -   The 'Close' prices are normalized to a range of (0, 1) using `MinMaxScaler`.
    -   Sequential data is created with a sequence length of 60 days (i.e., using the past 60 days' prices to predict the next day's price).
4.  **Data Splitting:**
    -   The data is split into training (80%) and testing (20%) sets.
    -   Input data `X` is reshaped to be suitable for LSTM input (samples, timesteps, features).
5.  **LSTM Model Building:**
    -   A `Sequential` Keras model is defined.
    -   It consists of two LSTM layers (50 units each) and a `Dense` output layer with 1 unit (for predicting the single 'Close' price).
6.  **Model Compilation:**
    -   The model is compiled using the 'adam' optimizer and 'mean_squared_error' as the loss function.
7.  **Model Training:**
    -   The LSTM model is trained on the training data for 20 epochs with a batch size of 32.
    -   Validation is performed on the test set during training.
8.  **Performance Visualization:**
    -   Training loss and validation loss are plotted over epochs to monitor for overfitting.
9.  **Prediction:**
    -   The trained model predicts stock prices on the test set.
    -   Predicted prices are inverse-transformed back to their original scale.
10. **Results Visualization:**
    -   Actual stock prices are plotted against the predicted stock prices for the test period.

## Setup and Usage

### Prerequisites

-   Python 3 (The notebook was run with Python 3.10.17, but other 3.x versions should work)
-   Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository (if applicable) or download the files.**
2.  **Install the required libraries:**
    You can install them using pip:
    ```bash
    pip install numpy pandas matplotlib seaborn yfinance scikit-learn tensorflow
    ```
    If you prefer to use a `requirements.txt` file, you can create one with the libraries listed above.

### Running the Notebook

1.  Navigate to the directory containing `stock.ipynb` in your terminal.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Open `stock.ipynb` from the Jupyter interface.
4.  Run the cells sequentially to execute the code.

## Results

The notebook produces two main visualizations:
1.  **Training vs. Validation Loss Plot:** This plot helps in assessing how well the model is learning and whether it's overfitting to the training data.
2.  **Stock Price Prediction Plot:** This plot compares the actual stock prices with the prices predicted by the LSTM model on the test data, providing a visual assessment of the prediction accuracy.

## Potential Issues

-   **CUDA/GPU Warnings:** The notebook output shows warnings related to CUDA drivers and cuDNN/cuBLAS factories not being found or already registered. This typically means a GPU is not being used for training, and TensorFlow will fall back to CPU. The code will still run correctly on a CPU.
-   `yfinance` API changes: `yf.download()` argument `auto_adjust` default has changed to `True`. This is informational and handled by the library.
