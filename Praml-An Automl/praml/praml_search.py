import os
import pandas as pd
from sklearn.pipeline import Pipeline
from .preprocessing import get_preprocessor
from .models import get_models
from .evaluation import evaluate_model, plot_learning_rate_distribution
from .utils import get_logger
from .tpe import TPE
from .hoo import HyperoptOptimizer
import numpy as np

logger = get_logger(__name__)

class PramlSearchResult:
    def __init__(self, best_model, best_model_name, best_score, best_score_hoo, best_pipeline, best_pipeline_hoo, best_params, ranklist, ranklist_hoo, best_model_name_hoo, best_params_hoo):
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.parameters = best_params
        self.ranklist = ranklist
        self.ranklist_hoo = ranklist_hoo
        self.best_model_name_hoo = best_model_name_hoo
        self.parameters_hoo = best_params_hoo
        self.best_score_hoo = best_score_hoo
        self.best_score = best_score
        self.best_pipeline = best_pipeline
        self.best_pipeline_hoo = best_pipeline_hoo

def praml_search(dataset, target_column, problem_type, basis, max_evals=50):
    logger.info("Starting praml_search")
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    text_features = X.select_dtypes(include=['string']).columns.tolist()
    time_features = X.select_dtypes(include=['datetime64']).columns.tolist()
    
    logger.debug("Preprocessing the dataset")
    preprocessor = get_preprocessor(numerical_features, categorical_features, text_features, time_features)
    models, param_grids = get_models(problem_type)
    
    pipelines = []
    for model_name, model in models:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipelines.append((model_name, pipeline))
    
    logger.info("Starting TPE optimization")
    tpe = TPE(pipelines, param_grids, max_evals)
    hoo = HyperoptOptimizer(pipelines, param_grids, max_evals)

    best_pipeline, best_params, best_score = tpe.optimize(X, y, problem_type)
    best_pipeline_hoo, best_params_hoo, best_score_hoo = hoo.optimize(X, y, problem_type)
    
    try:
        best_model_name = [name for name, model in pipelines if model == best_pipeline][0]
    except IndexError:
        logger.error("Best pipeline from TPE optimization not found in pipelines.")
        best_model_name = "Not Found"

    try:
        best_model_name_hoo = [name for name, model in pipelines if model == best_pipeline_hoo][0]
    except IndexError:
        logger.error("Best pipeline from HOO optimization not found in pipelines.")
        best_model_name_hoo = "Not Found"

    # Ranklist for TPE optimized pipeline
    ranklist_data = []
    learning_rates_tpe = []

    for model_name, model in pipelines:
        if tpe.best_scores[model_name] > -np.inf:
            score = tpe.best_scores[model_name]
            scores = evaluate_model(model, X, y, problem_type)
            ranklist_data.append((model_name, score, scores['f1_score'], scores['roc_auc'], scores['precision'], scores['recall'], scores.get('validation_score', np.nan)))
            if 'learning_rate' in scores:
                learning_rates_tpe.append(scores['learning_rate'])

    ranklist = pd.DataFrame(ranklist_data, columns=['Model', basis.capitalize(), 'F1 Score', 'ROC AUC', 'Precision', 'Recall', 'Validation Score'])
    ranklist = ranklist.sort_values(by=basis.capitalize(), ascending=False).reset_index(drop=True)
               
    # Ranklist for HOO optimized pipeline
    ranklist_data_hoo = []
    learning_rates_hoo = []
    for model_name, model in pipelines:
        if hoo.best_scores[model_name] > -np.inf:
            score = hoo.best_scores[model_name]
            scores = evaluate_model(model, X, y, problem_type)
            ranklist_data_hoo.append((model_name, score, scores['f1_score'], scores['roc_auc'], scores['precision'], scores['recall'], scores.get('validation_score', np.nan)))
            if 'learning_rate' in scores:
                learning_rates_hoo.append(scores['learning_rate'])
    
    ranklist_hoo = pd.DataFrame(ranklist_data_hoo, columns=['Model', basis.capitalize(), 'F1 Score', 'ROC AUC', 'Precision', 'Recall', 'Validation Score'])
    ranklist_hoo = ranklist_hoo.sort_values(by=basis.capitalize(), ascending=False).reset_index(drop=True)
        
    logger.info("praml_search completed")

    # Plot and save learning rate distribution for TPE
    results_dir = os.path.join(os.path.dirname(__file__), '../lr_curves_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plot_learning_rate_distribution(learning_rates_tpe, 'Learning Rate Distribution (TPE)', os.path.join(results_dir, 'learning_rate_distribution_tpe.png'))

    # Plot and save learning rate distribution for HOO
    plot_learning_rate_distribution(learning_rates_hoo, 'Learning Rate Distribution (HOO)', os.path.join(results_dir, 'learning_rate_distribution_hoo.png'))

    return PramlSearchResult(best_pipeline, best_model_name, best_score, best_score_hoo, best_pipeline, best_pipeline_hoo, best_params, ranklist, ranklist_hoo, best_model_name_hoo, best_params_hoo)
