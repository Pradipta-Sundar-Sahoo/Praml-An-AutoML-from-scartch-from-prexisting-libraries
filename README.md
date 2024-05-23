# praml: Python Automated Machine Learning Library

praml is a simple Python model for automated machine learning (AutoML). It simplifies the process of building machine learning models by automating data preprocessing, model selection, hyperparameter tuning, and model evaluation.

## Features

- Automates data preprocessing including feature scaling and one-hot encoding for categorical variables.
- Supports both classification and regression tasks.
- Provides a variety of built-in models including linear models, tree-based models, support vector machines, and ensemble techniques.
- Performs hyperparameter optimisation using own HPO implementation(Tree-prazen estimator) and hyperopt.
- Evaluates models based on customizable metrics such as accuracy, F1 score, recall, precision, and mean squared error.
- Returns the best model along with its parameters and a ranklist of models based on performance.

```bash
pip install -r requirements.txt
```
## Usage

Here's a basic example demonstrating how to use praml for automated machine learning:

```python
import pandas as pd
from praml import praml_search

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Specify the target column
target_column = 'target'

# Perform AutoML search for classification with accuracy
result = praml_search(dataset=df, target_column=target_column, problem_type="classification", basis="accuracy",max_evals=25)

# Print the best model and its parameters
print("Best classification model based on accuracy:", result.best_model)
print("Parameters of the best classification model:", result.parameters)
print("Ranklist based on accuracy by tpe:")
print(result.ranklist)
print("Ranklist based on accuracy by hoo:")
print(result.ranklist_hoo)
```

For more examples and detailed documentation, please refer to the praml documentation and explore the examples directory in the praml GitHub repository.

## Requirements

- Python (>=3.6)
- scikit-learn (>=0.24.2)
- numpy (>=1.21.0)
- pandas (>=1.3.0)
- xgboost (>=1.4.2)
- lightgbm (>=3.2.1)
- catboost (>=0.26)

## Key highlights
- integrated with various machine learning models for classification & regression and handle different data types(numerical,categorical, text and time_series data)
- ROC AUC, cross-validation, f1, accuracy, recall, precision scores are given in comparision with own HPO from scratch(TPE-optimisised) and pre-existing HPO library(hyperopt).
- Along with best_model_name, best_model's parameters can also be accessed.
- You can set different basis/metrics of choosing your model.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the praml GitHub repository.

## License

praml is licensed under the MIT License.

