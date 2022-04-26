import pandas as pd
import numpy as np
def correct_SCT_data(data, shuffle=False, seed=10):

    X = data
    X["region_east"] = 1.0 * (X["region"] == "East")
    X["region_west"] = 1.0 * (X["region"] == "West")
    X["region_north"] = 1.0 * (X["region"] == "North")
    X = X.drop(columns=["region", "moon_phase_name"])
    for col in X.columns:
        X[col] = X[col].astype(str).str.replace(',', '').astype(float)
        if X[col].isna().any():
            X[col] = X[col].fillna(0)
    if shuffle:
        np.random.seed(10)
        new_columns = np.array(X.columns)
        np.random.shuffle(new_columns)
        X = X[new_columns]
    return X


def correct_SCT_labels(labels):
    return labels.drop(columns=
                       ['attendance', 'number_over_4_hours',
                        'number_over_8_hours', 'percentage_within_8_hours',
                        'number_over_12_hours', 'percentage_within_12_hours',
                        'number_under_4_hours', 'number_4_hours_8_hours',
                        'number_8_hours_12_hours', 'percentage_within_4_hours'])


def correct_US_data(data, shuffle=False, seed=10):
    X = data
    for col in X.columns:
        X[col] = X[col].astype(str).str.replace(',', '').astype(float)
        if X[col].isna().any():
            X[col] = X[col].fillna(0)
    if shuffle:
        np.random.seed(10)
        new_columns = np.array(X.columns)
        np.random.shuffle(new_columns)
        X = X[new_columns]
    return X

def correct_US_labels(labels):
    return np.ravel(labels)

