import pandas as pd
import numpy as np


def common_var_checker(df_train, df_val, df_test, target):
    # Get the dataframe of common variables between the training, validation and test data
    df_common_var = pd.DataFrame(np.intersect1d(np.intersect1d(df_train.columns, df_val.columns), np.union1d(df_test.columns, [target])),
                                 columns=['common var'])

    return df_common_var


def id_checker(df, dtype='float'):
    # Get the dataframe of identifiers
    df_id = df[[var for var in df.columns
                # If the data type is not dtype
                if (df[var].dtype != dtype
                    # If the value is unique for each sample
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]

    return df_id


def datetime_transformer(df, datetime_vars):
    # The dictionary with key as datetime type and value as datetime type operator
    dict_ = {'year': lambda x: x.dt.year,
             'month': lambda x: x.dt.month,
             'day': lambda x: x.dt.day,
             'hour': lambda x: x.dt.hour,
             'minute': lambda x: x.dt.minute,
             'second': lambda x: x.dt.second}

    # Make a copy of df
    df_datetime = df.copy(deep=True)

    # For each variable in datetime_vars
    for var in datetime_vars:
        # Cast the variable to datetime
        df_datetime[var] = pd.to_datetime(df_datetime[var])

        # For each item (datetime_type and datetime_type_operator) in dict_
        for datetime_type, datetime_type_operator in dict_.items():
            # Add a new variable to df_datetime where:
            # the variable's name is var + '_' + datetime_type
            # the variable's values are the ones obtained by datetime_type_operator
            df_datetime[var + '_' +
                        datetime_type] = datetime_type_operator(df_datetime[var])

    # Remove datetime_vars from df_datetime
    df_datetime = df_datetime.drop(columns=datetime_vars)

    return df_datetime


def nan_checker(df):
    # Get the dataframe of variables with NaN, their proportion of NaN and data type
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(
        by='proportion', ascending=False).reset_index(drop=True)

    return df_nan


def cat_var_checker(df, dtype='object'):
    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           # If the data type is dtype
                           for var in df.columns if df[var].dtype == dtype],
                          columns=['var', 'nunique'])

    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(
        by='nunique', ascending=False).reset_index(drop=True)

    return df_cat
