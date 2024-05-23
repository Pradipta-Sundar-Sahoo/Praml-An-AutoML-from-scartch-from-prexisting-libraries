from sklearn.model_selection import cross_val_score
from .utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
logger = get_logger(__name__)

def plot_learning_rate_distribution(learning_rates, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_roc_curves(models, X, y, path):
    logger.debug("Plotting ROC curves")
    plt.figure()
    
    for name, model in models:
        model.fit(X, y)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()
    logger.debug("ROC curves plot saved to %s", path)

def evaluate_model(model, X, y, problem_type):
    logger.info("Evaluating model: %s", model)
    metrics = {}
    if problem_type == 'classification':
        accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()
        roc_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc_ovr').mean()
        precision = cross_val_score(model, X, y, cv=5, scoring='precision_weighted').mean()
        recall = cross_val_score(model, X, y, cv=5, scoring='recall_weighted').mean()
        metrics['accuracy'] = accuracy
        metrics['f1_score'] = f1
        metrics['roc_auc'] = roc_auc
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['validation_score']=np.nan
    elif problem_type == 'regression':
        mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        metrics['validation_score'] = mse
    else:
        logger.error("Invalid problem type: %s", problem_type)
        raise ValueError(f"Unknown problem type: {problem_type}")
    
     # Extract the learning rate if the model has it
    if hasattr(model.named_steps['model'], 'learning_rate'):
        metrics['learning_rate'] = model.named_steps['model'].learning_rate

    return metrics
