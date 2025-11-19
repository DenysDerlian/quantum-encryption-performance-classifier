# Importar as bibliotecas necessárias
try: 
    import os
    import kagglehub
    import shutil
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    import pandas as pd
    import numpy as np
    import numpy.typing as npt
    from typing import List, Tuple, Dict, Optional, Sequence
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from imblearn.over_sampling import RandomOverSampler
    from tensorflow.keras import layers, models, callbacks, optimizers, losses, regularizers
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import precision_recall_curve, accuracy_score
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    from scipy import stats as ss
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import optuna
    import os
    from pathlib import Path
    import json
    from optuna.samplers import TPESampler
    from optuna.visualization import plot_optimization_history, plot_param_importances
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder

except ImportError as e:
    print(f"Erro ao importar bibliotecas: {e}")
    print("Certifique-se de que todas as dependências estão instaladas.")
    raise e
    
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Verificar se o tensorflow está utilizando a GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))