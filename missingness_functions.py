import numpy as np
import pandas as pd 

# Given a dataframe ts_data, find the amount of percentage missingness in the non hadm_id columns
def calculate_tab_completeness(ts_data):
    if 'hadm_id' in ts_data.columns:
        completeness = 1 - ts_data.drop(columns=['hadm_id']).isnull().sum().sum() / (ts_data.shape[0] * ts_data.shape[1])
    else:
        completeness = 1 - ts_data.isnull().sum().sum() / (ts_data.shape[0] * ts_data.shape[1])
    return completeness

def remove_rows_until_complete(data, percentage):
    num_features = len(data.columns)

    # Calculate how many missing values are in each row in data 'missing_amount'
    data['filled_counts'] = data.notnull().sum(axis=1)
    data = data.sort_values(['filled_counts'], ascending=False)
    filled_df = data['filled_counts']
    while filled_df.sum() / (num_features * len(filled_df)) < percentage:
        filled_df = filled_df.iloc[:-1]

    return data.loc[filled_df.index].drop(columns=['filled_counts'])

def sparse_remove_data_until_percentage_complete(data, percentage, random_state):
    data_df = data.copy()

    num_features = len(data_df.columns) - 1 if 'filled_counts' in data_df.columns else len(data_df.columns)
    min_completeness = 1 / num_features

    if percentage < min_completeness:
        raise ValueError(f"Desired completeness {percentage:.2f} is less than the minimum possible completeness {min_completeness:.2f}.")

    data_df['filled_counts'] = data_df.notnull().sum(axis=1)

    # Calculate total number of values required for desired completeness
    total_values = data_df.shape[0] * num_features
    desired_filled_values = int(total_values * percentage)
    current_filled_values = data_df.drop(columns='filled_counts').notnull().sum().sum()
    values_to_remove = current_filled_values - desired_filled_values

    # Ensure each column retains at least one non-null value
    column_non_null_counts = data_df.drop(columns='filled_counts').notnull().sum()

    np.random.seed(random_state)
    removed_values = 0

    while removed_values < values_to_remove:
        # Select a random row with more than one filled value
        non_empty_rows = data_df[data_df['filled_counts'] > 1].index
        if non_empty_rows.empty:
            print("Breaking: No eligible rows remaining.")
            break

        random_row = np.random.choice(non_empty_rows)
        non_empty_cols = data_df.loc[random_row].dropna().index.difference(['filled_counts'])

        # Avoid removing the last value from any column
        eligible_cols = [col for col in non_empty_cols if column_non_null_counts[col] > 1]

        if not eligible_cols:
            print("Breaking: No eligible columns remaining.")
            break

        # Remove a random non-empty value from the selected row
        random_col = np.random.choice(eligible_cols)
        data_df.at[random_row, random_col] = np.nan
        data_df.at[random_row, 'filled_counts'] -= 1
        column_non_null_counts[random_col] -= 1

        removed_values += 1

    data_df.drop(columns='filled_counts', inplace=True)
    return data_df
    