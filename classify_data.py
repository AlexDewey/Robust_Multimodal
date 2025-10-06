import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score

def train_and_predict(model_name, X_train, Y_train, X_test, Y_test, random_state):
    model_dict = get_model_dict(random_state)
    param_grid = get_model_search_space()
    
    model = model_dict[model_name]
    param_dist = param_grid.get(model_name, {})

    # Compute class weights for balancing (only for supported models)
    if model_name in ["LR", "RF", "DT"]:

        # Convert DataFrame to Series (if single column)
        if isinstance(Y_train, pd.DataFrame):
            if Y_train.shape[1] == 1:
                Y_train = Y_train.iloc[:, 0]

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        model.set_params(class_weight=class_weight_dict)

    # XGBoost specific balancing
    if model_name == "XGB":
        pos_weight = (Y_train.value_counts()[0] / Y_train.value_counts()[1]) if Y_train.value_counts()[1] > 0 else 1
        model.set_params(scale_pos_weight=pos_weight)

    # Perform RandomizedSearchCV
    grid_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = best_model.predict(X_test)

    test_auroc = roc_auc_score(Y_test, test_probs)
    test_auprc = average_precision_score(Y_test, test_probs)
    test_f1 = f1_score(Y_test, test_preds)
    test_accuracy = accuracy_score(Y_test, test_preds)

    test_auprc = average_precision_score(Y_test, test_probs)

    return {"auroc": test_auroc, "f1_score": test_f1, "accuracy": test_accuracy, "model": best_model, "auprc": test_auprc}

def get_model_dict(random_state):
    model_dict = dict()

    model = LogisticRegression(max_iter=1000,
                            solver='liblinear')
    model_dict['LR'] = model

    model = xgb.XGBClassifier(loss='log_loss',
                            learning_rate=0.15,
                            random_state=random_state)
    model_dict['XGB'] = model

    model = RandomForestClassifier(class_weight='balanced',
                                n_estimators=75,
                                criterion='gini',
                                random_state=random_state)
    model_dict['RF'] = model

    model = DecisionTreeClassifier(class_weight='balanced',
                                criterion='entropy',
                                splitter='random',
                                random_state=random_state)
    model_dict['DT'] = model

    model = KNeighborsClassifier(n_neighbors=5)
    model_dict['KNN'] = model

    model = MLPClassifier(hidden_layer_sizes=(100, 100),
                            max_iter=1000,
                            random_state=random_state)
    model_dict['MLP'] = model

    model = GradientBoostingClassifier(n_estimators=75,
                                       random_state=random_state)
    model_dict['BT'] = model

    return model_dict

def get_model_search_space():
    # Randomized search hyperparameter configuration
    param_grid = {
        'XGB': {
            'learning_rate': np.linspace(0.01, 0.9, 10),
            'max_depth': [3, 5, 7, 9, 12, 15, 20],
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.5, 1.0],  
            'subsample': np.linspace(0.1, 1.0, 5),  
            'colsample_bytree': np.linspace(0.1, 1.0, 5), 
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
            'reg_lambda': [0, 1, 2, 5, 10],  
            'n_estimators': [50, 100, 200, 300],
            'tree_method': ['hist']
        },
        'RF': {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [3, 5, 7, 9, 12, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'LR': {
            'C': [0.1, 1, 10],
            'tol': [1e-4, 1e-3, 1e-2],
            'solver': ['liblinear', 'saga', 'newton-cholesky'],
            'max_iter': [1000]
        },
        'DT': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7, 9, 12, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'min_impurity_decrease': [0.0, 0.1, 0.2],
            'class_weight': ['balanced']
        },
        'KNN': {
            'n_neighbors': [1, 2, 3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2]
        },
        'MLP': {
            'hidden_layer_sizes': [(10,), (50,), (100,), (200,), (50, 50), (100, 100), (200, 200)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [3000]
        },
        'BT': {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': [3, 5, 7, 9, 12, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'loss': ['log_loss', 'exponential'],
            'learning_rate': np.linspace(0.01, 0.9, 10),

        }
    }
        
    return param_grid