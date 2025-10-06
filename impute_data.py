import pandas as pd
import numpy as np
import os
import subprocess

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

import MIDASpy as md

def custom_impute_df(df, imputation):  

    # If an entire column is NaN, fill it with zeros
    for i in range(df.shape[1]):  # iterate by column position, not name
        if df.iloc[:, i].isna().all():
            df.iloc[:, i] = 0

    if imputation == 'mean':
        # Mean imputation
        mean_imputation = df.fillna(df.mean())
        df = mean_imputation

    elif imputation == 'median':
        # Median imputation
        median_imputation = df.fillna(df.median())
        df = median_imputation

    elif imputation == 'knn':
        # KNN Imputer
        imputer = KNNImputer(n_neighbors=2)
        knn_imputation = imputer.fit_transform(df)
        knn_imputation = pd.DataFrame(knn_imputation, index=df.index, columns=df.columns)
        df = knn_imputation

    elif imputation == 'cart':
        # CART Imputer
        cart_imputer = SimpleImputer(strategy='constant', fill_value=np.nan)
        df_cart = cart_imputer.fit_transform(df)
        
        for column in df.columns:
            missing_rows = df[column].isna()
            if missing_rows.any():
                tree = DecisionTreeRegressor()
                not_missing_rows = ~missing_rows
                tree.fit(df.loc[not_missing_rows, df.columns != column], df.loc[not_missing_rows, column])
                df_cart[missing_rows, df.columns.get_loc(column)] = tree.predict(df.loc[missing_rows, df.columns != column])
        
        df_cart = pd.DataFrame(df_cart, index=df.index, columns=df.columns)
        df = df_cart

    elif imputation == 'mice-lr':
        # MICE with Linear Regression
        # Identify columns with all NaN values
        all_nan_cols = df.columns[df.isna().all()]
        non_nan_cols = df.columns[~df.isna().all()]
        
        if len(all_nan_cols) > 0:
            # Only impute columns with at least one non-NaN value
            df_to_impute = df[non_nan_cols]
            imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
            imputed_data = imputer.fit_transform(df_to_impute)
            
            # Create DataFrame with imputed data
            result_df = pd.DataFrame(imputed_data, index=df.index, columns=non_nan_cols)
            
            # Add back all-NaN columns
            for col in all_nan_cols:
                result_df[col] = np.nan
            
            # Return DataFrame with original column order
            df = result_df[df.columns]
        else:
            # If no all-NaN columns, proceed as before
            imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
            mice_lr_imputation = imputer.fit_transform(df)
            mice_lr_imputation = pd.DataFrame(mice_lr_imputation, index=df.index, columns=df.columns)
            df = mice_lr_imputation

    elif imputation == 'mice-dt':
        # MICE with Decision Tree
        imputer = IterativeImputer(estimator=DecisionTreeRegressor(random_state=0), random_state=0)
        mice_dt_imputation = imputer.fit_transform(df)
        mice_dt_imputation = pd.DataFrame(mice_dt_imputation, index=df.index, columns=df.columns)
        df = mice_dt_imputation

    elif imputation == 'random':
        # Randomly fill in missing values with other values from the same column
        for column in df.columns:
            if df[column].isna().any():
                non_missing_values = df[column].dropna().values
                random_values = np.random.choice(non_missing_values, size=df[column].isna().sum(), replace=True)
                df.loc[df[column].isna(), column] = random_values
    
    elif imputation == 'midas':

        scaler = MinMaxScaler()
        df_copy = df.copy()

        # Scale the DataFrame
        df_copy[df_copy.columns] = pd.DataFrame(scaler.fit_transform(df_copy[df_copy.columns]), columns=df_copy.columns)

        random_seed = 0
        imputer = md.Midas(layer_structure=[64], vae_layer=False, seed=random_seed, input_drop=1)
        
        # Build and train the model
        imputer.build_model(df_copy, softmax_columns=[])
        imputer.train_model(training_epochs=10)
        
        # Generate a single imputation
        df = imputer.generate_samples(m=1).output_list[0]
        df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
    
    elif imputation == 'zero':
        # Fill NaN values with 0
        df = df.fillna(0)
        
        
    return df