import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os


# Función para intentar convertir cada elemento a float
def convert_to_float(x):
    if isinstance(x, str):
        try:
            return float(x.replace(',', '.'))
        except ValueError:
            return x
    else:
        return x


def proportion_nan(df, want_print: bool):
    total_values = df.size
    nan_values = df.isna().sum().sum()

    if want_print:
        print(nan_values / total_values)

    return nan_values / total_values


def round_floats_to_2_decimals(np_array):
    return np.vectorize(lambda x: round(x, 2) if isinstance(x, float) else x)(np_array)


# Función modificada para verificar si el elemento es float, int o NaN

def check_type(x):
    return isinstance(x, (float, int)) or pd.isna(x)


def report_invalid_values(df, result):
    # Función para reportar valores no válidos en un DataFrame.

    # Parámetros:
    # df: DataFrame de pandas.
    # result: DataFrame de pandas con el resultado de aplicar una verificación de tipo a cada elemento de df.

    all_valid = result.all().all()
    if not all_valid:
        invalid_values = np.where(~result)  # Obtiene las posiciones de los valores no válidos
        for i, j in zip(*invalid_values):
            invalid_value = df.iloc[i, j]
            print(
                f"Valor no válido en fila {i}, columna '{df.columns[j]}': {invalid_value} (Tipo: {type(invalid_value)})")


def export_dataframe(df, filename_prefix):
    # Exporta un DataFrame a un archivo CSV,
    # incluyendo la fecha y hora actual en el nombre del archivo.

    # Parámetros:
    # df: DataFrame de pandas a exportar.
    # filename_prefix: Prefijo del nombre del archivo para el archivo CSV.

    # Obtener la fecha y hora actual
    now = datetime.now()
    # Formatear la fecha y hora para usar en el nombre del archivo
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Crear el nombre del archivo incluyendo la fecha y hora
    filename = f'output/{filename_prefix}_{formatted_now}.csv'
    # Guardar el DataFrame en un archivo CSV con el nombre generado
    df.to_csv(filename, sep=';', decimal=',', index=False)
    print(f'Archivo exportado: {filename}')


def export_lists_to_csv(list1, list2, filename_prefix):
    # Convert the lists to a DataFrame
    df = pd.DataFrame({
        'Datos Reales': list1,
        'Datos Imputados': list2,
    })

    # Get the current date and time
    now = datetime.now()
    # Format the date and time for use in the file name
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Create the file name including the date and time
    filename = f'output/{filename_prefix}_{formatted_now}.csv'
    # Save the DataFrame to a CSV file with the generated name
    df.to_csv(filename, sep=';', decimal=',', index=False)
    print(f'File exported: {filename}')


def data_analysis(df, filename_prefix='KNN_Imputed'):
    analysis_results = ""

    df_gonna_be_imputed_filtered = df.copy()
    df_gonna_be_original_filtered = df.copy()

    for column in df_gonna_be_imputed_filtered.columns:
        filtered_imputed_values = df_gonna_be_imputed_filtered[column].apply(
            lambda x: x if isinstance(x, float) and
                           len(str(x).split('.')[1]) == 3 else None).dropna()

        if not filtered_imputed_values.empty:

            max_value_imputed = filtered_imputed_values.max()
            min_value_imputed = filtered_imputed_values.min()
            mean_value_imputed = filtered_imputed_values.mean()

            variance_value = filtered_imputed_values.var()

            analysis_results += f'Análisis de los datos imputados de la columna {column}:\n'
            analysis_results += f'Valor máximo Imputado: {max_value_imputed: .3f}\n'
            analysis_results += f'Valor mínimo Imputado: {min_value_imputed: .3f}\n'
            analysis_results += f'Valor promedio Imputado: {mean_value_imputed: .3f}\n'
            analysis_results += f'Varianza de Datos Imputados: {variance_value: .3f}\n'
            analysis_results += '-------------------------------------\n'

        else:
            analysis_results += f'La columna {column} no tiene valores float con 3 decimales.\n'

    for column in df_gonna_be_original_filtered.columns:
        filtered_original_values = df_gonna_be_original_filtered[column].apply(
            lambda x: x if isinstance(x, float) and
                           len(str(x).split('.')[1]) < 3 else None).dropna()

        if not filtered_original_values.empty:

            max_value_original = filtered_original_values.max()
            min_value_original = filtered_original_values.min()
            mean_value_original = filtered_original_values.mean()

            variance_value = filtered_original_values.var()

            analysis_results += f'Análisis de datos originales de la columna {column}:\n'
            analysis_results += f'Valor máximo original: {max_value_original: .3f}\n'
            analysis_results += f'Valor mínimo original : {min_value_original: .3f}\n'
            analysis_results += f'Valor promedio original: {mean_value_original: .3f}\n'
            analysis_results += f'Varianza original : {variance_value: .3f}\n'
            analysis_results += '-------------------------------------\n'

        else:
            analysis_results += f'La columna {column} no tiene valores float con menos de 3 decimales.\n'

    # Obtener la fecha y hora actual
    now = datetime.now()
    # Formatear la fecha y hora para usar en el nombre del archivo
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Crear el nombre del archivo incluyendo la fecha y hora

    directory_path = 'output/dataAnalysis'
    filename = f'{filename_prefix}_{formatted_now}_DATAANALYSIS.txt'
    full_path = f'{directory_path}/{filename}'
    # Guardar los resultados en un archivo .txt con el nombre generado
    with open(full_path, 'w') as file:
        file.write(analysis_results)
    print(f'Análisis guardado en: {filename}')


def plot_variances(df, filename_prefix='KNN_Imputed'):
    # matplotlib.use('TkAgg') #

    for column in df.columns:
        filtered_values = df[column].apply(
            lambda x: x if isinstance(x, float) and
                           len(str(x).split('.')[1]) == 3 else None).dropna()

        if not filtered_values.empty:
            mean_value = filtered_values.mean()
            centered_values = filtered_values - mean_value

            # Calcular la distancia al origen para cada punto
            distances = np.abs(centered_values)

            # Crear un mapa de colores que va de verde a rojo
            color_map = cm.get_cmap('RdYlGn_r')

            # Asignar un color a cada punto en función de su distancia al origen
            colors = color_map(distances / distances.max())

            plt.scatter(range(len(centered_values)), centered_values, c=colors)
            plt.title(f'Gráfico de dispersión de la columna {column}')
            plt.xlabel('Índice')
            plt.ylabel('Valor')
            plt.grid(True)
            plt.show()
        else:
            print(f'La columna {column} no tiene valores float con 3 decimales.')


def extract_decimal_values(df1, df2):
    # Initialize lists to store the extracted values
    df1_values = []
    df2_values = []

    # Initialize a list to store the coordinates of the values with 3 decimals
    coordinates = []

    # Iterate over the first DataFrame
    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            # Check if the value is a float with 3 decimals
            if isinstance(df1.iloc[i, j], float) and len(str(df1.iloc[i, j]).split('.')[1]) == 3:
                # Save the value and its coordinates
                df1_values.append(df1.iloc[i, j])
                coordinates.append((i, j))

    # Iterate over the coordinates
    for coord in coordinates:
        # Extract the corresponding value from the second DataFrame
        df2_values.append(df2.iloc[coord])

    return df1_values, df2_values


def test_algorythm_accuracy(df, nan_porcentage: float):
    # We take a sample of 300 rows from the dataframe with all data is complete

    df_original = df.copy()
    df_original = df_original.drop(columns=['Numero Paciente'])

    mask = df_original.map(lambda x: isinstance(x, float) and len(str(x).split('.')[1]) < 3)
    df_training = df_original[mask.all(axis=1)]

    if len(df_training) >= 300:
        df_training = df_training.sample(n=300)

    df_test = df_training.copy()

    total_values = df_test.size
    nan_count = int(total_values * nan_porcentage)

    random_indexes = np.random.randint(0, total_values, nan_count)

    for index in np.nditer(random_indexes):
        row = index // df_test.shape[1]
        column = index % df_test.shape[1]
        df_test.iloc[row, column] = np.nan

    k_value = int(np.sqrt(len(round(df_test))))

    imputer = KNNImputer(n_neighbors=k_value, weights='distance')

    numpy_df = df_test.to_numpy()  # convert data to numpy array
    scaler = StandardScaler()

    numpy_df_scaled = scaler.fit_transform(numpy_df)  # scale data to mean 0 and variance 1
    vector_standarized_imputed = imputer.fit_transform(numpy_df_scaled)  # impute data
    df_imputed = pd.DataFrame(vector_standarized_imputed, columns=df_test.columns)  # convert numpy array to DataFrame

    # Invert Standarization

    numpy_df_imputed_scaled_back = scaler.inverse_transform(df_imputed)  # invert standarization
    df_imputed_scaled_back = pd.DataFrame(numpy_df_imputed_scaled_back, columns=df_test.columns)
    df_imputed_scaled_back = df_imputed_scaled_back.map(lambda x: round(x, 3))

    imputed_data, true_data = extract_decimal_values(df_imputed_scaled_back, df_training)

    print(f'Data Range: {max(true_data) - min(true_data)}')

    mae = mean_absolute_error(true_data, imputed_data)
    mse = mean_squared_error(true_data, imputed_data)
    r2 = r2_score(true_data, imputed_data)

    print(f'MAE: {mae} MSE: {mse} R2: {r2}')

    export_lists_to_csv(true_data, imputed_data, 'test/Imputed_vs_Original')
    export_dataframe(df_training, 'accuracy_test/Original_Training')
    export_dataframe(df_imputed_scaled_back, 'accuracy_test/Imputed_Training')
