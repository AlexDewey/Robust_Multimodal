import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy

from classify_data import train_and_predict
from impute_data import custom_impute_df
from missingness_functions import sparse_remove_data_until_percentage_complete

def normalize_distribution(values, bins=50):
    '''
    Helper function to estimate distributions for KLD.
    '''
    hist, _ = np.histogram(values, bins=bins, density=True)  # Estimate probability density
    hist += 1e-10  # Avoid zero probabilities
    return hist / hist.sum()

def analyze_downstream_imputations(dataframe,
                                   target_column,
                                   clf_model,
                                   iterations,
                                   complete_rates=None,
                                   imputers_given='all',
                                   nan_test_cols=None,
                                   n_splits=5):
    '''
    Evaluates the impact of various imputation methods on downstream classification performance 
    under different data completeness scenarios.

    Parameters:
        dataframe (pd.DataFrame): Input dataset with features and target column.

        target_column (str): Name of the target variable to predict.
        
        clf_model (str): Classifier identifier used in `train_and_predict()`.
        
        iterations (int): Number of experiment repetitions with different random seeds.
        
        complete_rates (list[float], or None): List of target completeness percentages (0-1). If done, imputations 
        first conduct mean imputation, remove values, then impute with selected methods.
        
        imputers_given (list[str] or str): List of imputer methods to evaluate or 'all' for default list.
        
        nan_test_cols (list[str], or None): Specific columns to set as NaN in test folds to validate unimodal to 
        multumodal imputation.
        
        n_splits (int): Number of cross-validation folds.
    
    Returns:
        pd.DataFrame: Aggregated results containing all experimentation metrics:
            "iteration"
            "complete_rate"
            "n_splits"
            "target_column"
            "nan_test_cols"
            "imputer"
            "classifier"
            "classifier_auroc"
            "classifier_auprc"
            "classifier_f1"
            "imputation_accuracy"
            "kld"
    '''

    if complete_rates is None:
        complete_rates = ['none']
    
    if imputers_given == 'all':
        imputers = ['mean', 'median', 'knn', 'cart', 'mice-lr', 'mice-dt', 'random', 'zero']
    else:
        imputers = imputers_given

    results = []

    for iteration in range(iterations):

        print(iteration) # Tracks progress

        random_state = iteration

        for complete_rate in complete_rates:

            data = dataframe.copy()
            Y = data[target_column]
            data.drop(columns=target_column, inplace=True)
            data = data.astype(float)

            for imputer in imputers:

                # Here we have to decide if we want a specific amount of completeness.
                # If we decide to have this, we impute mean first before removing a certain amount.
                # Otherwise, we go to our regular imputation.
                data_copy = data.copy()
                if complete_rate is not 'none':
                    data_copy = custom_impute_df(data_copy, 'mean')
                    df_ECM_w_missing = sparse_remove_data_until_percentage_complete(data=data_copy, percentage=complete_rate, random_state=random_state)
                else:
                    data_copy = custom_impute_df(data_copy, imputer)
                    df_ECM_w_missing = data_copy
                df_ECM_w_missing_copy = df_ECM_w_missing.copy()
                imputed_data = custom_impute_df(df_ECM_w_missing_copy, imputer)

                # Calculate RMSE
                mask = df_ECM_w_missing.isna()
                if mask.any().any():
                    imputed_values = imputed_data[mask].values.ravel()
                    true_values = data_copy[mask].values.ravel()
                    rmse = np.sqrt(mean_squared_error(true_values[~np.isnan(true_values)], 
                                                        imputed_values[~np.isnan(imputed_values)]))
                    
                    # Calculate KLD
                    true_dist = normalize_distribution(true_values[~np.isnan(true_values)])  
                    imputed_dist = normalize_distribution(imputed_values[~np.isnan(imputed_values)])  
                    kl_divergence = entropy(true_dist, imputed_dist)
                else:
                    rmse = np.nan
                    kl_divergence = np.nan

                # Perform Stratified K-Fold cross-validation
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                fold_scores = []

                for train_idx, test_idx in skf.split(imputed_data, Y):
                    if nan_test_cols is not None: # Set nan_cols to np.nan and impute
                        temp_data = imputed_data.copy()
                        # Set temp dataframes test_idx columns to np.nan
                        temp_data.iloc[test_idx, temp_data.columns.isin(nan_test_cols)] = np.nan
                        X_train, X_test = temp_data.iloc[train_idx], temp_data.iloc[test_idx]
                        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
                        combined_data = pd.concat([X_train, X_test], axis=0)
                        # This imputation does not bias the training data, so we can use it to impute the test data
                        combined_data = custom_impute_df(combined_data, imputer)
                        # Seperate X_test again to be sent into the training prediction function
                        X_train = combined_data.iloc[:len(X_train)]
                        X_test = combined_data.iloc[len(X_train):]
                    else:
                        X_train, X_test = imputed_data.iloc[train_idx], imputed_data.iloc[test_idx]
                        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

                    score = train_and_predict(
                        model_name=clf_model,
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        Y_test=Y_test,
                        random_state=random_state
                    )

                    fold_scores.append(score)

                # For each entry in fold_scores, calculate accuracy metrics
                auroc_value = 0
                auprc_value = 0
                f1_value = 0
                for idx, fold in enumerate(fold_scores):
                    auroc_value += fold['auroc']
                    auprc_value += fold['auprc']
                    f1_value += fold['f1_score']
                auroc_value /= len(fold_scores)
                auprc_value /= len(fold_scores)
                f1_value /= len(fold_scores)

                results_obj = {
                    "iteration": iteration,
                    "complete_rate": complete_rate,
                    "n_splits": n_splits,
                    "target_column": target_column,
                    "nan_test_cols": nan_test_cols,
                    "imputer": imputer,
                    "classifier": clf_model,
                    "classifier_auroc": auroc_avg,
                    "classifier_auprc": auprc_avg,
                    "classifier_f1": f1_avg,
                    "imputation_accuracy": rmse,
                    "kld": kl_divergence
                }

                # Append experiment results
                results.append(results_obj)

    # Convert results to a DataFrame
    return pd.DataFrame(results)
