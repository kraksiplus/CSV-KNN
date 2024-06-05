import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from helpers import (convert_to_float, export_dataframe, data_analysis, plot_variances, test_algorythm_accuracy,
                     proportion_nan)

df = pd.read_csv('KNN.csv', sep=';', decimal=',')
df = df.map(convert_to_float)  # convert data to float if is necessary

df_without_patient = df.drop('Numero Paciente', axis=1)
mask = df_without_patient.notna().all(axis=1)
df_training = df_without_patient[mask]
df_test = df_training.copy()
total_values = df_test.size
nan_count = int(total_values * 0.1)
random_indexes = np.random.randint(0, len(df_training), nan_count)

for index in np.nditer(random_indexes):
    row = index // df_test.shape[1]
    column = index % df_test.shape[1]
    df_test.iloc[row, column] = np.nan

k_value = int(np.sqrt(len(round(df_test))))
imputer = KNNImputer(n_neighbors=k_value, weights='distance')  # KNNImputer with DISTANCE weights
numpy_df_test = df_test.to_numpy()  # convert data to numpy array
numpy_df_training = df_training.to_numpy()  # convert data to numpy array

scaler = StandardScaler()

numpy_df_training_scaled = scaler.fit_transform(numpy_df_training)  # scale data to mean 0 and variance 1

numpy_df_test_scaled = scaler.fit_transform(numpy_df_test)  # scale data to mean 0 and variance 1

vector_test_standarized_imputed = imputer.fit_transform(numpy_df_test_scaled)  # impute data
vector_training_standarized_imputed = imputer.fit_transform(numpy_df_training_scaled)  # impute data

# -----------------------------------

knn_regresor = KNeighborsRegressor(n_neighbors=k_value, weights='distance')

# Fit the KNN regressor on the training data
knn_regresor.fit(numpy_df_training_scaled, numpy_df_training)

for i, uncompleted_row in enumerate(numpy_df_test_scaled):
    if not np.isnan(uncompleted_row).any():
        # Use the KNN regressor to predict the missing values in the test data
        completed_row = knn_regresor.predict([uncompleted_row])
        numpy_df_test_scaled[i] = completed_row

# Use the KNN regressor to make predictions on the test data
predictions = knn_regresor.predict(numpy_df_test_scaled[~np.isnan(numpy_df_test_scaled).any(axis=1)])

# Compare the predictions to the actual values
mse = mean_squared_error(numpy_df_test_scaled[~np.isnan(numpy_df_test_scaled).any(axis=1)], predictions)
mae = mean_absolute_error(numpy_df_test_scaled[~np.isnan(numpy_df_test_scaled).any(axis=1)], predictions)
r2 = r2_score(numpy_df_test_scaled[~np.isnan(numpy_df_test_scaled).any(axis=1)], predictions)

print(f"MAE: {mae} MSE: {mse} R2: {r2}")