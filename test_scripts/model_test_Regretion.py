import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             median_absolute_error, mean_squared_log_error, explained_variance_score)
from helpers import (convert_to_float, export_dataframe, data_analysis, plot_variances, test_algorythm_accuracy,
                     proportion_nan, extract_decimal_values, export_lists_to_csv)

df = pd.read_csv('../output/complete_data/Complete_data_sample_2024-06-05_08-28-08.csv',
                 sep=';', decimal=',')
df = df.map(convert_to_float)  # convert data to float if is necessary
df_with_nan = df.copy()

nan_count = 900
random_row_indexes = np.random.randint(0, len(df), nan_count)
random_column_indexes = np.random.randint(0, df.shape[1], nan_count)

for i in range(nan_count):
    df_with_nan.iloc[random_row_indexes[i], random_column_indexes[i]] = np.nan

hist_regresor = HistGradientBoostingRegressor()
scaler = StandardScaler()
df_nan_filled = df_with_nan.copy()
df_nan_filled[df_nan_filled.columns] = scaler.fit_transform(df_nan_filled[df_nan_filled.columns])

for column in df_nan_filled.columns[df_nan_filled.isna().any()].tolist():
    df_train = df_nan_filled[df_nan_filled[column].notna()]
    df_test = df_nan_filled[df_nan_filled[column].isna()]

    hist_regresor.fit(df_train.drop(column, axis=1), df_train[column])
    df_nan_filled.loc[df_test.index, column] = hist_regresor.predict(df_test.drop(column, axis=1))

df_nan_filled[df_nan_filled.columns] = scaler.inverse_transform(df_nan_filled[df_nan_filled.columns])

imputed_data, true_data = extract_decimal_values(df_nan_filled, df)

print(imputed_data, true_data)


export_dataframe(df_nan_filled, '../hist_gradient_regressor/HistGradientBoostingRegressor_imputed')
export_lists_to_csv(true_data, imputed_data, '../hist_gradient_regressor/data_compare')

mae = mean_absolute_error(true_data, imputed_data)
mse = mean_squared_error(true_data, imputed_data)
medae = median_absolute_error(true_data, imputed_data)
msle = mean_squared_log_error(true_data, imputed_data)
rmsle = np.sqrt(msle)
evs = explained_variance_score(true_data, imputed_data)


r2 = r2_score(true_data, imputed_data)

# Compare the imputed data with the original data
# mse = mean_squared_error(df.dropna(), df_nan_filled.dropna())
# mae = mean_absolute_error(df.dropna(), df_nan_filled.dropna())
# r2 = r2_score(df.dropna(), df_nan_filled.dropna())

print(f"MAE: {mae} MSE: {mse} MEDAE: {medae} MSLE: {msle} RMSLE: {rmsle} EVS: {evs}  R2: {r2}")
