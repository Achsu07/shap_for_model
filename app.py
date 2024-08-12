!pip install shap


import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib


training_df = pd.read_csv('X_train.csv')
misclassification_df = pd.read_csv('xgb_misclassified77_features 2.csv')

model = joblib.load('xgbtest_model.joblib')

explainer = shap.Explainer(model.predict_proba, training_df, model_output = 'probability')
shap_values = explainer(misclassification_df)