import numpy as np
from ...impute_data import custom_impute_df
from ...classify_data import train_and_predict

def large_training_splits(y_col, train_df, test_df, validate_df, classifier, imputation_method):

    X_train = train_df.drop(columns=y_col)
    Y_train = train_df[y_col].astype(int).squeeze()

    X_test = test_df.drop(columns=y_col)
    Y_test = test_df[y_col].values[0].astype(int).squeeze()

    X_validate = validate_df.drop(columns=y_col)
    Y_validate = validate_df[y_col].values[0].astype(int).squeeze()

    # Impute (either each modality or combined)
    X_train = custom_impute_df(df=X_train, imputation=imputation_method)
    X_test = custom_impute_df(df=X_test, imputation=imputation_method)
    X_validate = custom_impute_df(df=X_validate, imputation=imputation_method)

    # Train and predict
    _, _, y_pred_train = train_and_predict(model_name=classifier,
                                           model=model_dict[classifier],
                                           X_train=X_train,
                                           Y_train=Y_train,
                                           X_test=X_train,
                                           imputation=imputation_method)
    _, _, y_pred_test = train_and_predict(model_name=classifier,
                                          model=model_dict[classifier],
                                          X_train=X_train,
                                          Y_train=Y_train,
                                          X_test=X_test,
                                          imputation=imputation_method)
    _, _, y_pred_validate = train_and_predict(model_name=classifier,
                                              model=model_dict[classifier],
                                              X_train=X_train,
                                              Y_train=Y_train,
                                              X_test=X_validate,
                                              imputation=imputation_method)

    return y_pred_train, Y_train, y_pred_test, Y_test, y_pred_validate, Y_validate
    