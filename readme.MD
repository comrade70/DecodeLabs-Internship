# Car Price Prediction Using Machine Learning

## Project Overview
This project focuses on predicting car prices using **Machine Learning** with the **XGBoost Regressor** algorithm. The complete machine learning workflow includes:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Missing value handling
- Data preprocessing and feature encoding
- Model training and hyperparameter tuning
- Performance evaluation
- Prediction on unseen test data

The goal is to build a predictive model capable of predicting used vehicle prices based on features such as brand, model, mileage, fuel type, and model year.

---

## Dataset Description

The dataset contains vehicle-related attributes used to predict the target variable (**price**).

### Features Used
Some of the key features include:

- **brand** – Car manufacturer
- **model** – Vehicle model
- **model_year** – Year of manufacture
- **milage** – Distance travelled
- **fuel_type** – Type of fuel used
- **accident** – Accident history
- **clean_title** – Vehicle title condition

### Target Variable
- **price** – The selling price of the vehicle

---

## Project Workflow

### 1. Importing Required Libraries
The notebook uses the following Python libraries:

- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Data visualisation
- **Scikit-learn** – Model training, evaluation, and preprocessing
- **XGBoost** – Regression model for prediction

---

### 2. Data Loading
The training dataset is loaded into a Pandas DataFrame for analysis and preprocessing.

Example:
```python
train_df = pd.read_csv("datasets/train.csv")
```

---

### 3. Exploratory Data Analysis (EDA)
EDA is performed to understand:

- Dataset shape
- Data types
- Statistical summary
- Missing values
- Duplicate records
- Correlation between numerical variables

Visualisations such as histograms, heatmaps, scatterplot, and count plots are used to better understand feature distributions and relationships.

---

### 4. Data Cleaning and Preprocessing

#### Missing Value Handling
The notebook handles missing values intelligently:

- **fuel_type** values are filled using the mode grouped by **brand** and **model**
- Missing values in:
  - `accident` → replaced with `"unknown"`
  - `clean_title` → replaced with `"unknown"`

#### Categorical Encoding
Categorical features are transformed into numerical values using **Label Encoding**.

---

### 5. Train-Test Split
The dataset is divided into:

- **80% Training Data**
- **20% Validation Data**

This helps evaluate how well the model generalises to unseen data.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

### 6. Model Training
The project uses the **XGBoost Regressor (XGBRegressor)** for price prediction.

Model initialisation:

```python
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)
```

---

### 7. Hyperparameter Tuning
To improve performance, **RandomizedSearchCV** is used for hyperparameter optimization.

This helps identify the best combination of model parameters while reducing training time compared to grid search.

---

### 8. Model Evaluation
Model performance is evaluated using:

#### Root Mean Squared Error (RMSE)

RMSE measures the average prediction error magnitude.

Formula:

```text
RMSE = √(Mean Squared Error)
```

---

### 9. Prediction on Test Dataset
After training, the model is applied to unseen test data to generate predictions.

The predictions are saved into a submission DataFrame:

```python
submission = pd.DataFrame({
    "id": test["id"],
    "Price": y_test_pred
})
```

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualisation |
| Seaborn | Statistical Visualisation |
| Scikit-learn | ML Utilities |
| XGBoost | Regression Model |


---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/comrade70/DecodeLabs-Internship.git

cd DecodeLabs-Internship
```

---

## Outcome
The trained model predicts vehicle prices based on available car attributes and generates predictions for new datasets.

---

## Future Improvements
Potential improvements include:
- Feature importance analysis
- Deployment as a web application

