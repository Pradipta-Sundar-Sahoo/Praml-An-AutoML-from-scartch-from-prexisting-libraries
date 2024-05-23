from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .utils import get_logger

logger = get_logger(__name__)

def get_preprocessor(numerical_features, categorical_features, text_features=None, time_features=None):
    logger.debug("Creating preprocessors for numerical, categorical, text, and time series features")
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer())
    ])
    
    time_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    transformers = []
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    if text_features:
        transformers.append(('txt', text_transformer, text_features))
    if time_features:
        transformers.append(('time', time_transformer, time_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.debug("Preprocessors created successfully")
    return preprocessor
