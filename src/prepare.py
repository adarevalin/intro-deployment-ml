from dvc import api  # Importación de la API de DVC para gestionar datos versionados
import pandas as pd
from io import StringIO  # Para trabajar con datos de tipo cadena en memoria
import sys
import logging  # Para registro de eventos

# Configuración básica del registro de eventos (logging)
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',  # Formato del mensaje de registro
    level=logging.INFO,  # Nivel de registro: INFO (mostrar mensajes informativos)
    datefmt='%H:%M:%S',  # Formato del tiempo en el registro
    stream=sys.stderr  # Flujo de salida al que se enviarán los registros (consola)
)

logger = logging.getLogger(__name__)  # Instancia de logger específica para este script

logger.info('Obteniendo datos...')

try:
    # Lectura de archivos CSV desde el repositorio versionado con DVC
    movie_data_path = api.read('dataset/movies.csv', remote='dataset-track', encoding="utf8")
    finantial_data_path = api.read('dataset/finantials.csv', remote='dataset-track', encoding="utf8")
    opening_gross_data_path = api.read('dataset/opening_gross.csv', remote='dataset-track', encoding="utf8")

    # Convertir los datos de cadena leídos desde DVC en DataFrames de Pandas
    movie_data = pd.read_csv(StringIO(movie_data_path))
    fin_data = pd.read_csv(StringIO(finantial_data_path))
    opening_gross_data = pd.read_csv(StringIO(opening_gross_data_path))

    logger.info('Datos obtenidos con éxito.')

    # Seleccionar columnas numéricas del DataFrame de películas
    numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
    numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
    movie_data = movie_data[numeric_columns + ['movie_title']]

    # Seleccionar columnas relevantes del DataFrame financiero
    fin_data = fin_data[['movie_title', 'production_budget', 'worldwide_gross']]

    # Combinar datos financieros y datos de películas en un DataFrame único
    fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
    full_movie_data = pd.merge(opening_gross_data, fin_movie_data, on='movie_title', how='left')

    # Eliminar columnas innecesarias del DataFrame final
    full_movie_data = full_movie_data.drop(['gross', 'movie_title'], axis=1)

    # Guardar los datos finales en un archivo CSV
    full_movie_data.to_csv('dataset/full_data.csv', index=False)

    logger.info('Datos obtenidos y preparados con éxito.')

except UnicodeDecodeError as e:
    logger.error(f'Error de decodificación Unicode: {e}')
except Exception as e:
    logger.error(f'Ocurrió un error: {e}')

