# California Housing Data Analysis

This repository contains a Jupyter notebook that performs data analysis and visualization on the California Housing dataset. The analysis includes data preprocessing, feature engineering, and model building to predict housing prices.

## Table of Contents
- [Overview](#overview)
- [Feature Exploration](#feature-exploration)
- [Feature Engineering](#feature-engineering)
- [Feature Selection](#feature-selection)
- [Preprocessing](#preprocessing)
- [Pipelining](#pipelining)
- [Model Building](#model-building)
- [Validation Techniques](#validation-techniques)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview
The goal of this project is to predict the median house value in California districts using various features provided in the dataset. The notebook covers the following steps:
- Data loading and exploration
- Data  preprocessing and cleaning
- Feature engineering and selection
- Building and evaluating regression models
- Hyperparameter tuning
- Saving the final model

## Feature Exploration
We began by loading the dataset and performing an initial exploration to understand its structure and key statistics.

### Key Steps:
1. **Dataset Overview**: Viewing the first few rows, column information, and summary statistics.
2. **Visualizations**: Creating histograms for all numerical features to understand their distributions.

![enter image description here](https://i.ibb.co/z7cKBCB/Screenshot-2024-05-29-at-2-07-56-PM.png)

## Feature Engineering
Next, we created new features to enhance our model's predictive power. 

### Key Steps:
1. **Creating New Features**: 
   - `room_per_house` (total_rooms / households)
   - `bedrooms_ratio` (total_bedrooms / total_rooms)
   - `people_per_house` (population / households)

```python
housing['room_per_house'] = housing['total_rooms'] / housing['households']
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"] 
housing["people_per_house"] = housing["population"] / housing["households"]
```

## Feature Selection
We used correlation analysis to select the most relevant features for predicting the median house value.

### Key Steps:
1. **Correlation Matrix**: Calculating and visualizing the correlation between features and the target variable.

![enter image description here](https://i.ibb.co/3srzYKJ/Screenshot-2024-05-29-at-2-09-32-PM.png)

## Preprocessing
Data preprocessing involved handling missing values, encoding categorical features, and scaling numerical features.

### Key Steps:
1. **Imputation**: Replacing missing values with the median value of each column.
2. **Encoding**: Transforming categorical variables into numerical format using one-hot encoding.
3. **Scaling**: Standardizing numerical features.


```python
imputer = SimpleImputer(strategy='median')
housing_num = imputer.fit_transform(housing_num)

encoder = OneHotEncoder()
housing_cat = encoder.fit_transform(housing_cat)
```

## Pipelining
We created data pipelines to streamline the preprocessing steps, making the workflow more efficient and reproducible.

### Key Steps:
1. **Pipeline Creation**: Combining imputation, scaling, and encoding into a single pipeline for numerical and categorical data.
2. **Column Transformer**: Applying the appropriate transformations to different columns in the dataset.
```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessing = ColumnTransformer([
    ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
])
```

## Model Building
We built several regression models to predict the median house value and evaluated their performance.

### Key Steps:
1. **Linear Regression**
2. **Decision Tree Regression**
3. **Random Forest Regression**

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
```

## Validation Techniques
We employed various validation techniques to ensure our model's robustness.

### Key Steps:
1. **Cross-Validation**: Evaluating models using 10-fold cross-validation.
2. **Grid Search**: Tuning hyperparameters using grid search.


```python
from sklearn.model_selection import cross_val_score, GridSearchCV

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15], 'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

## Results
### Model Evaluation
The final model's performance was evaluated on the test set, and the RMSE was calculated.

### Key Steps:
1. **Final Model Evaluation**: Evaluating the best model from the grid search on the test set.
2. **Feature Importances**: Identifying the most important features used by the model.
```python
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)

feature_importances = final_model.named_steps["random_forest"].feature_importances_
sorted(zip(feature_importances, final_model.named_steps["preprocessing"].get_feature_names_out()), reverse=True)
```


## Acknowledgments
- The dataset is from the California Housing Prices dataset available on Kaggle.
- This project follows the hands-on example from the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.

