from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from .utils import get_logger

logger = get_logger(__name__)

def get_models(problem_type):
    logger.debug("Getting models for problem type: %s", problem_type)
    if problem_type == 'classification':
        models = [
            ('logreg', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier()),
            ('svc', SVC()),
            
            ('catboost', CatBoostClassifier(silent=True)),
            ('extratrees', ExtraTreesClassifier())
        ]
        param_grid = {
            'logreg': {
                'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'model__penalty': ['l1','l2', None],
                'model__solver': ['saga']
            },
            'rf': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            },
            'svc': {
                'model__C': [0.1, 1.0, 10.0, 100.0],
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model__gamma': ['scale', 'auto'],
                'model__degree': [2, 3, 4],
                'model__coef0': [0.0, 0.1, 0.5, 1.0]
            },
            'catboost': {
                'model__iterations': [100, 200, 500],
                'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'model__depth': [4, 6, 8, 10],
                'model__l2_leaf_reg': [1, 3, 5, 7],
                'model__bagging_temperature': [0.5, 1.0, 2.0]
            },
            'extratrees': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            }
        }
    elif problem_type == 'regression':
        models = [
            ('linreg', LinearRegression()),
            ('rf', RandomForestRegressor()),
            ('svr', SVR()),
            ('xgb', xgb.XGBRegressor()),
            ('catboost', CatBoostRegressor(silent=True)),
            ('extratrees', ExtraTreesRegressor())
        ]
        param_grid = {
            'linreg': {
                # No parameters for LinearRegression in this example
            },
            'rf': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            },
            'svr': {
                'model__C': [0.1, 1.0, 10.0, 100.0],
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model__gamma': ['scale', 'auto'],
                'model__degree': [2, 3, 4],
                'model__coef0': [0.0, 0.1, 0.5, 1.0]
            },
            'xgb': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__min_split_loss': {'dist': 'uniform', 'low': 0.0, 'high': 10.0},
                'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'model__max_depth': [3, 6, 9, 12],
                'model__subsample': [0.5, 0.7, 1.0],
                'model__colsample_bytree': [0.5, 0.7, 1.0]
            },
            'catboost': {
                'model__iterations': [100, 200, 500],
                'model__learning_rate': [0.01, 0.1, 0.2, 0.3],
                'model__depth': [4, 6, 8, 10],
                'model__l2_leaf_reg': [1, 3, 5, 7],
                'model__bagging_temperature': [0.5, 1.0, 2.0]
            },
            'extratrees': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            }
        }
    else:
        logger.error("Invalid problem type: %s", problem_type)
        raise ValueError("problem_type must be either 'classification' or 'regression'")
    
    logger.debug("Models and parameter grids have been created.")
    return models, param_grid
