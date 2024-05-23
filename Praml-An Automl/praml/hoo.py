import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
from .utils import get_logger

logger = get_logger(__name__)

class HyperoptOptimizer:
    def __init__(self, models, param_grids, max_evals=50):
        self.models = models
        self.param_grids = param_grids
        self.max_evals = max_evals
        self.best_scores = {model_name: -np.inf for model_name, _ in models}
        self.best_params = {model_name: None for model_name, _ in models}
        self.best_models = {model_name: None for model_name, _ in models}
        self.best_model_name = None

    def evaluate(self, model, X, y, params, problem_type):
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')
        return np.mean(scores)

    def objective(self, params):
        model_idx = params.pop('model')
        model_name, model = self.models[model_idx]
        model_params = {k.split('__', 1)[-1]: v for k, v in params.items()}

        valid_params = model.get_params().keys()
        model_params = {k: v for k, v in model_params.items() if k in valid_params}

        try:
            model.set_params(**model_params)
            scores = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error')
            score = np.mean(scores)
        except Exception as e:
            logger.error(f"Error with model {model_name} and params {model_params}: {e}")
            return {'loss': np.inf, 'status': STATUS_OK}

        if score > self.best_scores[model_name]:
            self.best_scores[model_name] = score
            self.best_models[model_name] = model
            self.best_params[model_name] = model_params
            # Track the overall best model across all evaluations
            if self.best_model_name is None or score > self.best_scores[self.best_model_name]:
                self.best_model_name = model_name

        return {'loss': -score, 'status': STATUS_OK}
    def suggest(self, l, g):
        gamma = 0.25
        n = len(l) + len(g)
        top_n = max(1, int(np.floor(gamma * n)))
        l_sorted = sorted(l, key=lambda x: x[1], reverse=True)
        g_sorted = sorted(g, key=lambda x: x[1], reverse=True)
        selected = np.random.choice(l_sorted[:top_n] + g_sorted[:top_n])
        return selected[0]

    def optimize(self, X, y, problem_type):
        self.X = X
        self.y = y
        self.problem_type = problem_type

        space = {
            'model': hp.choice('model', range(len(self.models)))
        }

        for model_name, param_grid in self.param_grids.items():
            for param, values in param_grid.items():
                param_name = f'{model_name}__{param}'
                if isinstance(values, list):
                    space[param_name] = hp.choice(param_name, values)
                elif isinstance(values, dict):
                    if 'dist' in values and values['dist'] == 'uniform':
                        space[param_name] = hp.uniform(param_name, values['low'], values['high'])
                    elif 'dist' in values and values['dist'] == 'normal':
                        space[param_name] = hp.normal(param_name, values['loc'], values['scale'])
        
        trials = Trials()
        fmin(self.objective, space=space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)

        best_model_name = self.best_model_name
        return self.best_models[best_model_name], self.best_params[best_model_name], self.best_scores[best_model_name]
