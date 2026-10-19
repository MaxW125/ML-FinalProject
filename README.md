# Irrigation Need Prediction

Final project for Intro to Machine Learning at Texas State. The goal is to predict how much irrigation a field needs (Low, Medium or High) from its soil properties, weather and crop data. The data comes from Kaggle's Playground Series S6E4.

## What's in the notebook

1. **EDA:** missing values, class balance, feature distributions, correlations, and how each feature relates to the target.
2. **Preprocessing:** label-encoding the categorical features and the target.
3. **Models:** starting simple and working up. A decision tree and logistic regression first, then an MLP neural network, then LightGBM, XGBoost and CatBoost.
4. **Feature engineering:** new features like rainfall × humidity and temperature ÷ rainfall, since hot, dry conditions should mean more irrigation.
5. **Tuning:** randomized search over Random Forest hyperparameters.
6. **Ensemble:** a soft-voting classifier that combines logistic regression and LightGBM.
7. **Submission:** training the ensemble on the full training set and writing predictions for the test set.

Every model is scored with 5-fold stratified cross-validation on balanced accuracy, so the comparison holds up even if the classes aren't evenly split.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost jupyter
jupyter notebook irrigation_prediction.ipynb
```

The data is in `TrainTestData/` (`train.csv`, `test.csv`, `sample_submission.csv`).
