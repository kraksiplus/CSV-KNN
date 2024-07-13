import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from helpers import (convert_to_float, export_dataframe, data_analysis, plot_variances, test_algorythm_accuracy,
                     proportion_nan)


df = pd.read_csv('KNN.csv', sep=';', decimal=',')
df = df.map(convert_to_float)  # convert data to float if is necessary
numpy_df = df.to_numpy()  # convert data to numpy array
scaler = StandardScaler()
numpy_df_scaled = scaler.fit_transform(numpy_df)  # scale data to mean 0 and variance 1
k_value = int(np.sqrt(len(round(df))))

# imputer = KNNImputer(n_neighbors=k_value, weights='uniform') # KNNImputer with UNIFORM weights
imputer = KNNImputer(n_neighbors=k_value, weights='distance')  # KNNImputer with DISTANCE weights

vector_standarized_imputed = imputer.fit_transform(numpy_df_scaled)
df_imputed = pd.DataFrame(vector_standarized_imputed, columns=df.columns)

# Invert Standarization
numpy_df_imputed_scaled_back = scaler.inverse_transform(df_imputed)
df_imputed_scaled_back = pd.DataFrame(numpy_df_imputed_scaled_back, columns=df.columns)
df_imputed_scaled_back = df_imputed_scaled_back.map(lambda x: round(x, 3))

# Export Imputed DataFrame to CSV

test_algorythm_accuracy(df_imputed_scaled_back, nan_porcentage=proportion_nan(df, want_print=False))
data_analysis(df_imputed_scaled_back)
export_dataframe(df_imputed_scaled_back, 'KNN_imputed') # explore imputed data to CSV file
# plot_variances(df_imputed_scaled_back, 'KNN_imputed') # plot variances of imputed data










