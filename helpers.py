import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
                             mean_squared_log_error, explained_variance_score, root_mean_squared_error)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import matplotlib
import os


# Function convert string number to float if is necessary
def convert_to_float(str_number):
    if isinstance(str_number, str):
        try:
            return float(str_number.replace(',', '.'))
        except ValueError:
            return str_number
    else:
        return str_number


# Measure the proportion of NaN values in a DataFrame
def proportion_nan(df, want_print: bool):
    total_values = df.size
    nan_values = df.isna().sum().sum()

    if want_print:
        print(nan_values / total_values)

    return nan_values / total_values


def round_floats_to_2_decimals(np_array):
    return np.vectorize(lambda x: round(x, 2) if isinstance(x, float) else x)(np_array)


# For testing purposes, check element type
def check_type(number):
    return isinstance(number, (float, int)) or pd.isna(number)


# For testing purposes, report invalid values
def report_invalid_values(df, result):
    all_valid = result.all().all()
    if not all_valid:
        invalid_values = np.where(~result)  # Obtiene las posiciones de los valores no válidos
        for i, j in zip(*invalid_values):
            invalid_value = df.iloc[i, j]
            print(
                f"Valor no válido en fila {i}, columna '{df.columns[j]}': {invalid_value} (Tipo: {type(invalid_value)})")


def export_dataframe(df, filename_prefix):
    # Export data frame to CSV file with the name of the file including the current date and time

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'output/{filename_prefix}_{formatted_now}.csv'
    df.to_csv(filename, sep=';', decimal=',', index=False)
    print(f'File Exported: {filename}')


def export_lists_to_csv(list1, list2, filename_prefix):
    # Create DataFrame from two lists to export to CSV for comparison between real and imputed data
    # with the same format of export_dataframe function

    df = pd.DataFrame({
        'Datos Reales': list1,
        'Datos Imputados': list2,
    })

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'output/{filename_prefix}_{formatted_now}.csv'
    df.to_csv(filename, sep=';', decimal=',', index=False)
    print(f'File exported: {filename}')


def export_variables(variables, filename_prefix):
    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'output/{filename_prefix}_{formatted_now}.txt'

    with open(filename, 'w') as file:
        for name, value in variables.items():
            file.write(f'{name}: {value}\n')
    print(f'Variables exportadas a: {filename}')


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

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    directory_path = 'output/dataAnalysis'
    filename = f'{filename_prefix}_{formatted_now}_DATAANALYSIS.txt'
    full_path = f'{directory_path}/{filename}'
    # Guardar los resultados en un archivo .txt con el nombre generado
    with open(full_path, 'w') as file:
        file.write(analysis_results)
    print(f'Analysis exported in: {filename}')


def plot_variances(df, filename_prefix='KNN_Imputed'):
    # matplotlib.use('TkAgg') #

    for column in df.columns:
        filtered_values = df[column].apply(
            lambda x: x if isinstance(x, float) and
                           len(str(x).split('.')[1]) == 3 else None).dropna()

        if not filtered_values.empty:
            mean_value = filtered_values.mean()
            centered_values = filtered_values - mean_value

            distances = np.abs(centered_values)
            color_map = cm.get_cmap('RdYlGn_r')
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
            if isinstance(df1.iloc[i, j], float) and len(str(df1.iloc[i, j]).split('.')[1]) >= 3:
                # Save the value and its coordinates
                df1_values.append(df1.iloc[i, j])
                coordinates.append((i, j))

    # Iterate over the coordinates
    for coord in coordinates:
        # Extract the corresponding value from the second DataFrame
        df2_values.append(df2.iloc[coord])

    return df1_values, df2_values

def count_three_decimals_values(df):
    three_decimals_count = {}
    percentage_three_decimals = {}

    for column in df.columns:
        count = df[column].apply(lambda x: isinstance(x, float) and len(str(x).split('.')[1]) >= 3).sum()
        three_decimals_count[column] = count
        percentage_three_decimals[column] = count / len(df[column])

    print(percentage_three_decimals)
    #print (three_decimals_count)
    # print("number of columns: ", len(percentage_three_decimals))
    return percentage_three_decimals


def export_three_decimal_counts(counts, filename_prefix):
    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'output/{filename_prefix}_{formatted_now}.txt'

    with open(filename, 'w') as file:
        for column, count in counts.items():
            file.write(f'{column}: {count}\n')
    print(f'Three decimal counts exported to: {filename}')



def test_algorythm_accuracy(df, nan_porcentage: dict):
    df_original = df.copy()
    print("Largo DF ORIGINAL: " ,len(df_original))

    try:
        df_original = df_original.drop(columns=['Numero Paciente'])
    except:
        pass

    mask = df_original.map(lambda x: isinstance(x, float) and len(str(x).split('.')[1]) < 3)
    df_training = df_original[mask.all(axis=1)]

    print("Largo DF TRAINING: " ,len(df_training))

    if len(df_training) >= 300:
        df_training = df_training.sample(n=300)

    df_test = df_training.copy()

    for column, percentage in nan_porcentage.items():
        total_values = len(df_test[column])
        nan_count = int(total_values * percentage)
        random_indexes = np.random.choice(df_test.index, nan_count, replace=False)
        df_test.loc[random_indexes, column] = np.nan

        # Verificación del porcentaje de NaN introducido
        actual_nan_count = df_test[column].isna().sum()
        actual_percentage = actual_nan_count / total_values
        print(f"Columna: {column}, Porcentaje esperado: {percentage}, Porcentaje real: {actual_percentage}")

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
    medae = median_absolute_error(true_data, imputed_data)
    msle = mean_squared_log_error(true_data, imputed_data)
    rmse = root_mean_squared_error(true_data, imputed_data)
    rmsle = np.sqrt(msle)
    evs = explained_variance_score(true_data, imputed_data)
    r2 = r2_score(true_data, imputed_data)

    car_dict = {
        'mae': mae, 'mse': mse, 'medae': medae,
        'msle': msle, 'rmsle': rmsle, 'evs': evs, 'r2': r2
    }

    # Calculate MSE for each column and count imputed values
    mse_per_column = {}
    imputed_counts = {}
    for column in df_training.columns:
        mse_per_column[column] = mean_squared_error(df_training[column], df_imputed_scaled_back[column])
        imputed_counts[column] = df_test[column].isna().sum()

    # Export MSE per column
    export_variables(mse_per_column, 'mse_per_column/mse_per_column')

    # Export imputed counts per column
    export_variables(imputed_counts, 'mse_per_column/imputed_counts_per_column')

    export_variables(car_dict, 'console_output/output')

    print(f"MAE: {mae} MSE: {mse} MEDAE: {medae} MSLE: {msle} RMSE: {rmse} RMSLE: {rmsle} EVS: {evs}  R2: {r2}")

    export_lists_to_csv(true_data, imputed_data, 'test/Imputed_vs_Original')
    export_dataframe(df_training, 'accuracy_test/original/Original_Training')
    export_dataframe(df_imputed_scaled_back, 'accuracy_test/imputed/Imputed_Training')