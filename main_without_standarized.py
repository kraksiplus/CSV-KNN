import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from helpers import convert_to_float, export_dataframe, data_analysis, plot_variances, round_floats_to_2_decimals

df = pd.read_csv('KNN.csv', sep=';', decimal=',')
print(len(df.columns))
df = df.map(convert_to_float) # convert data to float if is necessary
numpy_df = df.to_numpy() # convert data to numpy array

# df_standardized = pd.DataFrame(numpy_df_scaled, columns=df.columns)

k_value = int(np.sqrt(len(round(df))))

#imputer = KNNImputer(n_neighbors=k_value, weights='uniform') # KNNImputer with UNIFORM weights
imputer = KNNImputer(n_neighbors=k_value, weights='distance') # KNNImputer with DISTANCE weights

vector_imputed = imputer.fit_transform(numpy_df)
vector_imputed = round_floats_to_2_decimals(vector_imputed) # se redondean los valores de mas de 1 decimal a 2 decimales

# print(df_standardized_imputed)

df_imputed = pd.DataFrame(vector_imputed, columns=df.columns)
data_analysis(df_imputed)

print (df_imputed.iloc[:, :3].head(4))


# How first rows looks like
# print(df_imputed_scaled_back.iloc[:, :3].head(4))

# Export Imputed DataFrame to CSV

export_dataframe(df_imputed, 'KNN_imputed')
plot_variances(df_imputed, 'KNN_imputed')
