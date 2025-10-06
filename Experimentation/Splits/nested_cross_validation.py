def nested_cross_validation(df, y_col, model_dict, imputation_methods, modalities, groupings, missing_modality=None):
    # LOOCV-outer
    outer_predictions = list()
    outer_true_values = list()

    loo_out = LeaveOneOut()
    for train_index_out, test_index_out in loo_out.split(self.df):
        train_data_out = self.df.iloc[train_index_out]
        test_data_out = self.df.iloc[test_index_out]

        X_train_out = train_data_out.drop(columns=self.y_col)
        Y_train_out = train_data_out[self.y_col].astype(int).squeeze()

        X_test_out = test_data_out.drop(columns=self.y_col)
        Y_test_out = test_data_out[self.y_col].values[0].astype(int).squeeze()

        if missing_modality is not None:
            X_test_out[self.modalities[missing_modality]] = np.nan


        pred_list_in = list()
        true_list_in = list()

        # LOOCV-inner
        loo_in = LeaveOneOut()
        for train_index_in, test_index_in in loo_in.split(train_data_out):

            train_data_in = train_data_out.iloc[train_index_in]
            test_data_in = train_data_out.iloc[test_index_in]

            X_train_in = train_data_in.drop(columns=self.y_col)
            Y_train_in = train_data_in[self.y_col].astype(int).squeeze()

            X_test_in = test_data_in.drop(columns=self.y_col)
            Y_test_in = test_data_in[self.y_col].values[0].astype(int).squeeze()

            if missing_modality is not None:
                X_test_in[self.modalities[missing_modality]] = np.nan

            # Parallelize and train multiple models (either each modality its own model or combined)
            y_pred_dict = {method: {} for method in self.imputation_methods}
            for imputation in self.imputation_methods:

                X_train_in_copy = X_train_in.copy()
                Y_train_in_copy = Y_train_in.copy()
                X_test_in_copy = X_test_in.copy()

                # Impute (either each modality or combined)
                X_train_in_copy = custom_impute_df(df=X_train_in_copy, imputation=imputation)

                for model_name in self.model_dict:

                    ensemble_votes = list()
                    for group in groupings:
                        X_train_in_copy_group = X_train_in_copy[group]
                        X_test_in_copy_group = X_test_in_copy[group]
                        _, _, y_pred = train_and_predict(model_name=model_name,
                                                                        model=self.model_dict[model_name],
                                                                        X_train=X_train_in_copy_group,
                                                                        Y_train=Y_train_in_copy,
                                                                        X_test=X_test_in_copy_group,
                                                                        imputation=imputation)
                        ensemble_votes.append(y_pred)
                    
                    y_pred = np.round(np.mean(ensemble_votes, axis=0)).astype(int)
                    y_pred_dict[imputation][model_name] = y_pred
            
            # Record accuracies of each model in dictionary
            pred_list_in.append(y_pred_dict)
            true_list_in.append(Y_test_in)
