🏠 Real Estate House Price Prediction

👩‍💻 Authors

Maria, Hieuu, Saif

📋 Project Overview

This project predicts house sale prices using historical real estate data from 2001–2020.
It applies multiple machine learning regression models — including Linear Regression, Lasso Regression, Random Forest, and XGBoost — to identify the most accurate predictor for property prices.

The goal is to build a model that can estimate a property’s sale price based on its assessed value, type, location, and time of sale.

📂 Dataset

Source: Real Estate Sales Dataset (Kaggle)

Period: 2001–2020

Target Variable: Sale Amount

Key Features:

Assessed Value

Residential Type

Town

Year Sold

Property Age

Month Sold

Sales Ratio

Town Average Price

(Engineered features like is_summer, is_spring, etc.)

🧹 Data Preprocessing

Handled missing or invalid data (e.g., empty town names or property types)

Converted date columns to datetime format

Added new time-based and location-based features:

Years Until Sold, Season Flags (Spring/Summer/Winter)

Town Average Sale Price using group-based aggregation

Encoded categorical variables using LabelEncoder

Split data into train/test sets

🔍 Exploratory Data Analysis (EDA)

Distribution of property sale prices

Correlation heatmap between numerical features

Boxplots and histograms for outlier detection

Average price trends by year, town, and residential type

Key insights:

Assessed value is the strongest predictor of sale price.

Sale prices generally increased over time.

Certain towns show consistently higher median prices.

⚙️ Model Development

The following regression models were trained and compared:

Linear Regression

Lasso Regression

Random Forest Regressor

XGBoost Regressor

Each model was evaluated using RMSE, MAE, and R² metrics, along with cross-validation to ensure generalization.

🧠 Model Evaluation & Results
Model	CV RMSE	Description
Linear Regression	$0.00 (±$0.00)	Perfect fit (R² = 1.0). Indicates possible data leakage or highly correlated inputs.
Lasso Regression	$2.93 (±$0.65)	Adds regularization; interpretable and stable.
Random Forest	$22,617.17 (±$9,272.33)	Handles nonlinear patterns, moderate performance.
XGBoost	$24,748.44 (±$6,194.32)	Robust model but slightly overfitting on this dataset.

✅ Best Model: Linear Regression
💾 Saved As: best_house_price_model_linear_regression.pkl

🧩 Note: Linear Regression achieved perfect performance (R² = 1.0).
This suggests that some engineered features (like price_per_assessed or value_difference) may contain information derived from the target.
Removing them would produce a more realistic model performance (R² ≈ 0.8–0.9).

💡 Key Metrics Explained

RMSE (Root Mean Squared Error): Measures how far predictions are from actual prices — smaller = better.

MAE (Mean Absolute Error): Average absolute difference between predicted and actual prices.

R² (Coefficient of Determination): Measures how well the model explains variance (1.0 = perfect).

🧾 Sample Prediction

Input:

Town: Farmington
Residential Type: Single Family
Assessed Value: 507,500
Year Sold: 2021


Output:

Actual Price: $880,000
Predicted Price: $880,000
Prediction Error: $0.00

💾 Saving and Reusing the Model

The trained model was saved using Joblib:

import joblib
joblib.dump(best_model, 'best_house_price_model_linear_regression.pkl')


This allows quick reuse for future predictions without retraining.

🧩 Tools & Libraries

Python 3.10+

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

XGBoost

Joblib       


🧭 Future Improvements

Remove potential data leakage by excluding price_per_assessed and value_difference

Tune hyperparameters for XGBoost and Random Forest

Add SHAP or LIME for feature interpretability

Build a Streamlit or Flask web app for live predictions
