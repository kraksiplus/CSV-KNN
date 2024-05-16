import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from helpers import convert_to_float, export_dataframe, data_analysis

df = pd.read_csv('KNN.csv', sep=';', decimal=',')
print(len(df.columns))

df = df.map(convert_to_float) # convert data to float if is necessary

numpy_df = df.to_numpy() # convert data to numpy array

scaler = StandardScaler()
numpy_df_scaled = scaler.fit_transform(numpy_df) # scale data to mean 0 and variance 1
print(len(numpy_df_scaled[0]))

print(numpy_df_scaled)

# df_standardized = pd.DataFrame(numpy_df_scaled, columns=df.columns)

k_value = int(np.sqrt(len(round(df))))

#imputer = KNNImputer(n_neighbors=5, weights='uniform')
imputer = KNNImputer(n_neighbors=k_value, weights='distance')

vector_standarized_imputed = imputer.fit_transform(numpy_df_scaled)

#print(df_standardized_imputed)

df_imputed = pd.DataFrame(vector_standarized_imputed, columns=df.columns)

print (df_imputed.iloc[:, :3].head(4))

# Invert Standarization

numpy_df_imputed_scaled_back = scaler.inverse_transform(df_imputed)

df_imputed_scaled_back = pd.DataFrame(numpy_df_imputed_scaled_back, columns=df.columns)

df_imputed_scaled_back = df_imputed_scaled_back.map(lambda x: round(x, 2))

data_analysis(df_imputed_scaled_back)


# Mostrar las primeras filas para verificar
#print(df_imputed_scaled_back.iloc[:, :3].head(4))

# Exportar el DataFrame con los datos imputados y escalados a un archivo CSV

export_dataframe(df_imputed_scaled_back, 'KNN_imputed')
