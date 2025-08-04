# Airbnb Listing Popularity Prediction

Predicting **monthly reviews** for NYC Airbnb listings using a Gradient Boosting regressor.  
Best model achieves **Test R² = 0.54** and **Test MSE = 1.01** on a held-out split.

> **Highlights**
> - Predicted monthly reviews for NYC listings with **Gradient Boosting** achieving **test R² = 0.54** and **MSE = 1.01**  
> - Tuned model via **RandomizedSearchCV** to outperform **Ridge**, **Decision Tree**, and **Random Forest**  
> - Compared **cross-validation** and **test** metrics in tables to select best hyperparameters  
> - Interpreted feature impacts with **SHAP**, indicating **review count**, **recency (days since last review)**, and **price** as key drivers

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
- [Interpretability (SHAP)](#interpretability-shap)
- [Reproducibility](#reproducibility)


## Overview
This project builds a supervised regression pipeline to estimate `reviews_per_month` for Airbnb listings in New York City. We use robust preprocessing with `ColumnTransformer`, evaluate several baseline and tree-based models, tune hyperparameters via `RandomizedSearchCV`, and interpret the final model with SHAP.

## Dataset
- **Source**: Kaggle — *AB_NYC_2019* (Airbnb New York City 2019)  
  (Search for “AB_NYC_2019” on Kaggle.)
- **Target**: `reviews_per_month`  
- **Example features** (subset):
  - Numerical: `price`, `minimum_nights`, `availability_365`, `calculated_host_listings_count`, engineered logs
  - Categorical: `neighbourhood_group`, `neighbourhood`, `room_type` (one-hot encoded)
  - Temporal/engineered: `days_since_last_review`, `adjusted_price_per_night = price / (availability_365 + 1)`

> Note: Rows with missing target values are dropped in training/evaluation.

## Method
- **Split**: `train_test_split(test_size=0.3, random_state=123)`
- **Preprocessing**:
  - Numeric imputation + scaling
  - Categorical one-hot encoding
  - Optional binning for skewed variables
  - All steps encapsulated in a single **`Pipeline`**
- **Models compared**:
  - `DummyRegressor` (baseline)
  - `Ridge`
  - `DecisionTreeRegressor`
  - `RandomForestRegressor`
  - **`GradientBoostingRegressor` (best)**
- **Tuning**: `RandomizedSearchCV` with repeated trials per model; model selection by CV score and confirmed on test split.
- **Metrics**:
  - Primary: **MSE** (and RMSE where helpful)
  - Secondary: **R²**

## Results
| Model                    | CV R² (mean) | CV MSE (mean) | Test R² | Test MSE |
|-------------------------|--------------|---------------|---------|----------|
| Ridge                   | ~0.41        | ~1.41         | 0.42    | 1.27     |
| Decision Tree           | ~0.45        | ~1.31         | 0.50    | 1.09     |
| Random Forest           | ~0.51        | ~1.16         | 0.52    | 1.05     |
| **Gradient Boosting**   | **~0.52**    | **~1.15**     | **0.54**| **1.01** |

- **Selected model**: `GradientBoostingRegressor(n_estimators≈25, max_depth≈5, learning_rate≈0.15)`  
- The tuned Gradient Boosting model consistently outperforms linear and tree baselines on both CV and test.

## Interpretability (SHAP)
- **Top drivers** (global):  
  1) Overall review volume (e.g., `number_of_reviews`),  
  2) **Recency** of reviews (e.g., `days_since_last_review`),  
  3) **Price** level and its log/adjusted variants.
- SHAP summary/force plots indicate that listings with **recent activity** and **moderate pricing** tend to have higher predicted monthly reviews.


## Reproducibility
- Fixed `random_state=123` where applicable.
- End-to-end preprocessing + modeling done with a single **`Pipeline`**, ensuring identical transforms at train/test time.
- Evaluation uses a **held-out test split** (no CV leakage).

## Project Structure
