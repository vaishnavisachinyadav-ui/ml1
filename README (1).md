# 🏠 Simple House Price Prediction Using Linear Regression

An end-to-end beginner-friendly Machine Learning project demonstrating **Simple Linear Regression** to predict median house prices in California districts from **Median Income**, using the **California Housing Dataset** and **Scikit-Learn**.

**B.Tech – AI & Data Science | Practical Activity**

---

## 📋 Overview

This project implements a Simple Linear Regression model to predict median house values (`MedHouseVal`) across California block groups. Using a single input feature, **median income (`MedInc`)**, the model learns the straight-line relationship between the income of an area and its house prices, and is evaluated using standard regression metrics (MAE, MSE, RMSE and R²).

---

## 📊 Dataset: California Housing

The dataset is loaded directly via `sklearn.datasets.fetch_california_housing` (based on the 1990 U.S. Census). It contains **20,640 records** with **8 input features** and **1 continuous target variable**. There are no missing values and no duplicate records.

Dataset source: [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

### Features

| Feature | Description |
| :--- | :--- |
| **`MedInc`** | Median income in block group (in tens of thousands of USD) ✅ *used in this project* |
| **`HouseAge`** | Median house age in block group (years) |
| **`AveRooms`** | Average number of rooms per household |
| **`AveBedrms`** | Average number of bedrooms per household |
| **`Population`** | Block group population |
| **`AveOccup`** | Average number of household members |
| **`Latitude`** | Block group latitude coordinate |
| **`Longitude`** | Block group longitude coordinate |

### Target Variable

* **`MedHouseVal`**: Median house value for California districts, expressed in hundreds of thousands of dollars (\$100,000s).

### Why Median Income?

* Income is the **strongest single predictor** of house price (correlation ≈ **0.688**).
* Simple Linear Regression uses only **one** input feature, so the most useful one is chosen.

---

## 🧠 Methodology: Simple Linear Regression

Simple Linear Regression models the linear relationship between one independent variable ($X$) and a dependent target variable ($y$):

$$y = \beta_0 + \beta_1 X + \epsilon$$

Where:
* $y$ is the predicted median house value (`MedHouseVal`)
* $\beta_0$ is the intercept
* $\beta_1$ is the coefficient (slope)
* $X$ is the median income (`MedInc`)
* $\epsilon$ represents the residual error

The model minimizes the **Mean Squared Error (MSE)** via Ordinary Least Squares (OLS) to find the best-fitting straight line.

---

## 📈 Project Pipeline

The workflow in [`house_price_prediction.py`](./house_price_prediction.py) and [`House_Price_Prediction_Linear_Regression.ipynb`](./House_Price_Prediction_Linear_Regression.ipynb) follows these 10 steps:

1. **Dataset Selection**: Loading the dataset with `fetch_california_housing(as_frame=True)` and describing its source and features.
2. **Data Inspection**: First 5 records, shape, column names, data types, `df.describe()`, missing values and duplicates (handled with `dropna()` and `drop_duplicates()`).
3. **Feature Selection**: `X` = `MedInc`, `y` = `MedHouseVal`.
4. **Data Visualization**: Scatter plot of Median Income vs Median House Value.
5. **Train-Test Split**: Splitting the 20,640 records into training (80%, 16,512 samples) and testing (20%, 4,128 samples) sets using `train_test_split(..., test_size=0.2, random_state=42)`.
6. **Model Training**: Fitting `LinearRegression()` on `X_train` and `y_train`; displaying coefficient, intercept and equation.
7. **Prediction**: Predicting on `X_test` and showing actual vs predicted values side by side.
8. **Model Evaluation**: Calculating MAE, MSE, RMSE and R² on the test data only.
9. **Actual vs Predicted Visualization**: Scatter plot with a red dashed line for perfect predictions.
10. **New Prediction**: Predicting the house value for a new income value within the training range.

---

## 📊 Performance Results

Evaluated on the test split (20% unseen data):

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `0.6299` | Average prediction error of about **\$62,991** |
| **Mean Squared Error (MSE)** | `0.7091` | Average squared difference between actual and predicted values |
| **Root Mean Squared Error (RMSE)** | `0.8421` | Typical error size in normal units (about **\$84,209**) |
| **Coefficient of Determination ($R^2$)** | `0.4589` | The model explains **~45.89%** of the variance in California median house values |

### Model Parameters

* **Intercept ($\beta_0$)**: `0.4446`
* **Coefficient ($\beta_1$)**: `0.4193`

#### Regression Equation

$$\text{MedHouseVal} = 0.4193 \times \text{MedInc} + 0.4446$$

| Parameter | Value | Impact & Direction |
| :--- | :--- | :--- |
| **`MedInc` (coefficient)** | `+0.4193` | Strong positive impact; every \$10,000 increase in median income raises the predicted house value by about **\$41,900** |
| **Intercept** | `0.4446` | Starting point of the line (predicted value when income is 0; mathematical value only) |

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Machine Learning**: `scikit-learn`
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`
* **Development Environment**: Google Colab / Jupyter Notebook (`.ipynb`) / VS Code

---

## 📦 Getting Started

### 1. Prerequisites & Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn notebook
```

### 2. Run the Python Script

```bash
python house_price_prediction.py
```

Two graph windows will open. **Close each window to continue.** Graphs are also saved in the `outputs/` folder. (Internet is needed the first time to download the dataset.)

### 3. Run the Notebook

```bash
jupyter notebook House_Price_Prediction_Linear_Regression.ipynb
```

Or upload the notebook to [Google Colab](https://colab.research.google.com) and click **Runtime → Run all**.

### 4. Quick Code Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load California Housing Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Select Feature and Target
X = df[['MedInc']]
y = df['MedHouseVal']

# 3. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Prediction for New Input
new_data = pd.DataFrame({'MedInc': [5.0]})   # $50,000 median income

predicted_price = model.predict(new_data)
print(f"Predicted Median House Value: ${predicted_price[0] * 100000:,.2f}")
# Output: ~$254,128.98 (2.541 in $100k units)
```

---

## ⚠️ Limitations

1. Only **one feature** is used; location, house age, rooms and others are ignored.
2. House values are **capped at \$500,000** in the dataset.
3. The relationship is not perfectly linear.
4. The data is from **1990**, so it does not reflect current prices.
5. Predictions are for **area-level medians**, not individual houses.

**Possible improvement:** Use Multiple Linear Regression with all 8 features for better accuracy.

---

## 📄 License

This project is created for academic and educational purposes.
