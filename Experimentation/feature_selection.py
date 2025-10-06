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
from impute_data import custom_impute_df
from sklearn.model_selection import StratifiedKFold
import shap
from collections import defaultdict

def train_and_predict(model_name, imputation_method, X_train, Y_train, X_test, Y_test, random_state, grid_search_iterations):
    # Impute training data
    X_train = custom_impute_df(X_train, imputation=imputation_method)

    model_dict = get_model_dict(random_state)
    param_grid = get_model_search_space()
    
    model = model_dict[model_name]
    param_dist = param_grid.get(model_name, {})

    # Compute class weights for balancing (only for supported models)
    if model_name in ["LR", "RF", "DT"]:
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
        n_iter=grid_search_iterations,
        scoring='roc_auc',
        cv=5,
        random_state=random_state,
        n_jobs=4,
        refit=True
    )

    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Combine train and test for imputation
    combined = pd.concat([X_train, X_test], axis=0)
    combined_imputed = custom_impute_df(combined, imputation=imputation_method)
    # Get only the test data
    X_test = combined_imputed.iloc[len(X_train):]

    # Evaluate on imputed test set
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = best_model.predict(X_test)

    test_auroc = roc_auc_score(Y_test, test_probs)
    test_auprc = average_precision_score(Y_test, test_probs)
    test_f1 = f1_score(Y_test, test_preds)
    test_accuracy = accuracy_score(Y_test, test_preds)

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

def select_features(dataframe, target_column, R1=5, R2=5, n_splits=5, top_n_features=10):
    """
    Select features based on SHAP values from the best model across multiple iterations.
    """

    CLASSIFIERS = ['LR', 'RF', 'BT', 'KNN', 'XGB', 'MLP']
    IMPUTERS = ['mean', 'median', 'knn', 'cart', 'mice-lr', 'mice-dt']

    dataframe.astype(float)

    top_feature_subsets = []
    shap_means_for_correlations = []

    # Step 1: Variable Selection
    for R_1 in range(R1):

        print(R_1)
        
        data = dataframe.copy()
        Y = data[target_column]
        data.drop(columns=[target_column], inplace=True)

        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=R_1)

        shap_summaries = []

        for train_idx, test_idx in outer_cv.split(dataframe, Y): # V Fold

            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            shap_scores = []

            scores = {}

            for clf in CLASSIFIERS: # K Fold

                for imp in IMPUTERS:

                    # This computes our inner cross-validated score
                    result = train_and_predict(model_name=clf,
                                            imputation_method=imp,
                                            X_train=X_train,
                                            Y_train=Y_train,
                                            X_test=X_test,
                                            Y_test=Y_test,
                                            random_state=0,
                                            grid_search_iterations=R2)
                    scores[(clf, imp)] = [result['auroc'], result['model']]
            
            best_combo = max(scores.items(), key=lambda x: x[1][0])[0]
            best_model = scores[best_combo][1]
            best_classifier, best_imputation = best_combo

            # Combine train and test for imputation
            X_train_temp = X_train.copy()
            X_test_temp = X_test.copy()
            combined = pd.concat([X_train_temp, X_test_temp], axis=0)
            combined_imputed = custom_impute_df(combined, imputation=best_imputation)
            # Get only the test data
            X_train_R = combined_imputed.iloc[:len(X_train)]
            X_train_R = X_train_R.astype(float)

            if best_classifier in ['LR']:
                explainer = shap.Explainer(best_model, X_train_R)
                shap_values = explainer(X_train_R)
            elif best_classifier in ['KNN']:
                explainer = shap.KernelExplainer(best_model.predict_proba, X_train_R)
                shap_values = explainer(X_train_R)
            else:
                explainer = shap.TreeExplainer(best_model, X_train_R)
                shap_values = explainer(X_train_R, check_additivity=False)
            shap_abs = np.abs(shap_values.values).mean(axis=0)
            # If shab_abs is 2D, take first column
            if len(shap_abs.shape) > 1:
                shap_abs = shap_abs[:, 0]
            shap_scores.append(shap_abs)

            shap_mean = np.mean(shap_scores, axis=0)
            shap_summaries.append(shap_mean)

        # Get top n features based on SHAP values
        shap_summaries = np.array(shap_summaries)
        shap_mean_over_splits = np.mean(shap_summaries, axis=0)
        shap_means_for_correlations.append(shap_mean_over_splits) # This is for correlations only
        feature_importance = pd.Series(shap_mean_over_splits, index=data.columns)
        feature_importance.sort_values(ascending=False, inplace=True)
        top_features = feature_importance.head(top_n_features).index.tolist()
        top_feature_subsets.append(top_features)
    
    # Find features that are highly correlated, and remove the less impactful features
    # We conduct this at the end, as two highly correlated values can swap back and forth
    # in terms of importance, diminishing how often both make top placements.
    corr_matrix = dataframe.corr()
    shap_means_for_correlations_avg = np.mean(shap_means_for_correlations, axis=0)
    X_features = dataframe.drop(columns=target_column)
    shap_means_for_correlations_avg = pd.DataFrame([shap_means_for_correlations_avg], columns=X_features.columns)
    threshold = 0.8
    high_corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    similar_columns = high_corr_pairs[high_corr_pairs['Correlation'].abs() >= threshold]
    features_to_drop = set()
    for _, row in similar_columns.iterrows():
        col1, col2 = row['Feature1'], row['Feature2']
        
        # Skip if one is already marked for drop
        if col1 in features_to_drop or col2 in features_to_drop:
            continue

        # Compare SHAP scores
        def get_int_value(d, key):
            val = d.get(key, 0)
            return int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)
        shap1 = get_int_value(shap_means_for_correlations_avg, col1)
        shap2 = get_int_value(shap_means_for_correlations_avg, col2)
        if shap1 >= shap2:
            features_to_drop.add(col2)
        else:
            features_to_drop.add(col1)

    # The final subset are features that are selected more than 50% of the times
    # across all R1 iterations. 
    k_winner_feature_subset = (
        pd.Series([feat for subset in top_feature_subsets for feat in subset])
        .value_counts()
        .loc[lambda x: x >= (R1 // 2)]
        .index.tolist())

    # Drop highly correlated, poorer data    
    filtered_features = [f for f in k_winner_feature_subset if f not in features_to_drop]
    k_winner_feature_subset = filtered_features

    data = dataframe[k_winner_feature_subset + [target_column]]

    return k_winner_feature_subset

def select_model(dataframe, target_column, R1=5, n_splits=5):
    """
    Select the best model and imputation method based on cross-validated AUROC scores.
    """

    CLASSIFIERS = ['LR', 'RF', 'BT', 'KNN', 'BT']
    IMPUTERS = ['mean', 'median', 'knn', 'cart', 'mice-lr', 'mice-dt']

    scores = defaultdict(list)

    dataframe.astype(float)

    # Step 2: Classification Model Building
    for R_1 in range(R1):
        
        data = dataframe.copy()
        Y = data[target_column]
        data.drop(columns=[target_column], inplace=True)

        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=R_1)

        for train_idx, test_idx in outer_cv.split(dataframe, Y):

            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            for clf in CLASSIFIERS:

                for imp in IMPUTERS:

                    # This computes our inner cross-validated score
                    result = train_and_predict(model_name=clf,
                                            imputation_method=imp,
                                            X_train=X_train,
                                            Y_train=Y_train,
                                            X_test=X_test,
                                            Y_test=Y_test,
                                            random_state=R_1,
                                            grid_search_iterations=10)
                    scores[(clf, imp)].append(result['auroc'])

    # The best combo is the one with the highest mean AUROC
    best_combo = max(scores.items(), key=lambda x: np.mean(x[1]))[0]

    return best_combo
