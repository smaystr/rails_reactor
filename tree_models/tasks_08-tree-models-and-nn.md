# Homework 8

1. Extend the `/api/v1/statistics` and provide detail information about collected dataset
    - number of apartments
    - number of params
    - mean and std for price, square, year and etc

2. Generate automatic report for the dataset via [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) library
   Save it as a separate file and commit to MR.

3. Train DecisionTree (sklearn) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/)/[XGBoost](https://xgboost.readthedocs.io/en/latest/)/[CatBoost](https://github.com/catboost/catboost)(choose any) for price prediction.

4. Train Feed Forward Neural Network (pytorch) for price prediction. Consider the modification of the following hyperparams:
- activation function
- number of layers
- number of neurons per layer

5. Add `/api/v1/price/predict` endpoint for model inference. Endpoint accepts information about apartment in raw format and model type. All the preprocessing done inside enpoint. For example, get request to `/api/v1/price/predict` with params `{'model': 'decision_tree', features: {'area': 64,5, 'floor': 1, 'num_of_rooms': 2, ...}}` and the response is predicted price.

6. All the results should be reported in `report.md` file and committed in MR. The report should contain:
    - statistics from `/api/v1/statistics` endpoint
    - metric(MAE/MSE), loss and inference time for DecisionTree and LightGBM models
    - metric(MAE/MSE), loss and inference time for Feed-Forward NN for each set of hyperparams you tried
    - all hyperparams (K-fold/Cross-validation, size of train and test datasets, percentage for holdout)

## Bonus points:
  - add blending for boosting and neural network models. Add the results to `report.md`. Is there any improvements?

# Deadline

**Due on 16.08.2019 23:59**
