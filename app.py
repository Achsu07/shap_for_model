import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib


training_df = pd.read_csv('X_train.csv')
misclassification_df = pd.read_csv('xgb_misclassified77_features 2.csv')

model = joblib.load('xgbtest_model.joblib')

explainer = shap.Explainer(model.predict_proba, training_df, model_output = 'probability')
shap_values = explainer(misclassification_df)


def plot_shap_waterfall(index):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    titles = ["Benign", "Likely Benign", "Likely Pathogenic", "Pathogenic"]
    
    for i, title in enumerate(titles):
        shap.plots.waterfall(shap_values[index][i], max_display=10, show=False, ax=axes[i])
        axes[i].set_title(title)

    plt.tight_layout()
    return fig

# Streamlit app
st.title("SHAP Waterfall Plots")

# Input: Index of the sample to visualize
index = st.number_input("Enter index of the sample to visualize:", min_value=0, max_value=len(shap_values)-1, value=0)

# Generate and display the plot
fig = plot_shap_waterfall(index)
st.pyplot(fig)
