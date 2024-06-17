import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from joblib import dump

def update_model(model: Pipeline) -> None:
    """
    Save the given model pipeline to a file.
    
    Parameters:
    model (Pipeline): The machine learning pipeline to be saved.
    """
    dump(model, 'model/model.pkl')

def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    """
    Save a simple metrics report and the model pipeline description to a text file.
    
    Parameters:
    train_score (float): The training score of the model.
    test_score (float): The test score of the model.
    validation_score (float): The validation score of the model.
    model (Pipeline): The machine learning pipeline.
    """
    with open('report.txt', 'w') as report_file:
        report_file.write('# Model Pipeline Description\n')
        for key, value in model.named_steps.items():
            report_file.write(f'### {key}: {value.__repr__()}\n')
        report_file.write(f'## Train Score: {train_score}\n')
        report_file.write(f'## Test Score: {test_score}\n')
        report_file.write(f'## Validation Score: {validation_score}\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
    """
    Generate and save a scatter plot of real vs. predicted values to visualize model performance.
    
    Parameters:
    y_real (pd.Series): The real target values.
    y_pred (pd.Series): The predicted target values by the model.
    """
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    ax.set_title('Model Prediction Performance')
    fig.savefig('prediction.png')

# Example usage:
# Assume we have a trained model pipeline named 'model', and scores:
# model = Pipeline([...])
# train_score = 0.9
# test_score = 0.85
# validation_score = 0.87
# y_real = pd.Series([...])
# y_pred = pd.Series([...])

# update_model(model)
# save_simple_metrics_report(train_score, test_score, validation_score, model)
# get_model_performance_test_set(y_real, y_pred)
