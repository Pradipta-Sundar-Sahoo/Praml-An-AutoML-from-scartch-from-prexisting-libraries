import numpy as np
from sklearn.model_selection import cross_val_score
from .evaluation import evaluate_model
class TPE:
    def __init__(self, models, param_grids, max_evals=50):
        self.models = models
        self.param_grids = param_grids
        self.max_evals = max_evals
        self.best_scores = {model_name: -np.inf for model_name, _ in models}
        self.best_params = {model_name: None for model_name, _ in models}
        self.best_models = {model_name: None for model_name, _ in models}

    def sample_params(self, param_grid):
        sampled_params = {}
        for param, values in param_grid.items():
            if isinstance(values, list):
                sampled_params[param] = np.random.choice(values)
            elif isinstance(values, dict):
                if 'dist' in values and values['dist'] == 'uniform':
                    sampled_params[param] = np.random.uniform(values['low'], values['high'])
                elif 'dist' in values and values['dist'] == 'normal':
                    sampled_params[param] = np.random.normal(values['loc'], values['scale'])
        return sampled_params

    def suggest(self, l, g):
        gamma = 0.25
        n = len(l) + len(g)
        top_n = max(1, int(np.floor(gamma * n)))
        l_sorted = sorted(l, key=lambda x: x[1], reverse=True)
        g_sorted = sorted(g, key=lambda x: x[1], reverse=True)
        selected = np.random.choice(l_sorted[:top_n] + g_sorted[:top_n])
        return selected[0]

    def evaluate(self, model, X, y, params, problem_type):
        model.set_params(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if problem_type == 'classification' else 'neg_mean_squared_error')
        return np.mean(scores)

    def optimize(self, X, y, problem_type):
        for _ in range(self.max_evals):
            for model_name, model in self.models:
                param_grid = self.param_grids[model_name]
                params = self.sample_params(param_grid)
                score = self.evaluate(model, X, y, params, problem_type)
                if score > self.best_scores[model_name]:
                    self.best_scores[model_name] = score
                    self.best_models[model_name] = model
                    self.best_params[model_name] = params
        best_model_name = max(self.best_scores, key=self.best_scores.get)
        return self.best_models[best_model_name], self.best_params[best_model_name], self.best_scores[best_model_name]

