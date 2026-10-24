# Battery Temperature Prediction

A small Python project that estimates battery temperature from driving and battery measurements. It compares two regression models implemented from scratch with NumPy: **linear regression** and **degree-2 polynomial regression**. A Matplotlib dashboard displays training cost and test-set predictions, and lets you enter feature values to request predictions from both models.

## Dataset

The data comes from Kaggle's [Battery and Heating Data in Real Driving Cycles](https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles) dataset by `atechnohazard`.

The program expects a semicolon-separated CSV with these columns:

| Role | Dataset column |
| --- | --- |
| Input feature | `Battery Voltage [V]` |
| Input feature | `Battery Current [A]` |
| Input feature | `Ambient Temperature [°C]` |
| Input feature | `Time [s]` |
| Prediction target | `Battery Temperature [°C]` |

Download the dataset from Kaggle and place its CSV at `data/battery_data.csv`. The filename and capitalization must match because `main.py` reads that exact path. The repository currently includes `data/Battery_data.csv`; on case-sensitive systems, rename it to `battery_data.csv` before running the program.

## Models and workflow

1. The selected columns are converted to numeric NumPy arrays.
2. The rows are shuffled using seed `42` and split into 80% training and 20% test data.
3. Input features are standardized with z-scores using the **training set's** mean and standard deviation; those same values are applied to the test set and interactive inputs.
4. Both models are trained for 2,000 iterations with batch gradient descent and a learning rate of `0.05`.
   - **Linear regression** learns an intercept and a coefficient for each input feature.
   - **Polynomial regression (degree 2)** adds squared terms and pairwise interaction terms for the standardized inputs, then fits them using the same linear regression implementation.
5. Test-set performance is summarized with root mean squared error (RMSE). The dashboard also plots the training cost (`MSE / 2`) over iterations and actual versus predicted target values.

The polynomial feature expansion is applied after standardizing the original features. The target temperature is kept in its original units (°C). No specific model accuracy is promised; results depend on the dataset file and its contents.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt` (`numpy`, `pandas`, and `matplotlib`)

## Installation and usage

Clone or download this repository, ensure the dataset is at `data/battery_data.csv`, then run the following from the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and start the program:

```bash
python -m pip install -r requirements.txt
python main.py
```

The program prints the two test RMSE values and opens a Matplotlib window. Enter battery voltage, battery current, ambient temperature, and time, then click **Predict Both**. The dashboard displays predictions from both models.

## Project structure

```text
.
├── main.py                      # Data loading, training, comparison, and dashboard
├── requirements.txt             # Python dependencies
├── data/
│   └── Battery_data.csv          # Kaggle dataset (rename as described above)
└── src/
    ├── Preprocessing.py          # Feature selection, standardization, train/test split
    ├── Linear_Regression.py      # Linear regression with batch gradient descent
    └── polynomial_regression.py  # Degree-2 feature expansion and regression
```

## Data attribution

Please follow Kaggle's terms and the dataset creator's licensing and attribution requirements when using or redistributing the data. This project does not include a separate dataset license.
