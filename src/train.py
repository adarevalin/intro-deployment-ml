# Importación de bibliotecas necesarias
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate  # Funciones para dividir datos, búsqueda en malla y validación cruzada
from sklearn.pipeline import Pipeline  # Clase para crear un pipeline de procesamiento y modelo
from sklearn.impute import SimpleImputer  # Clase para imputación simple de datos faltantes
from sklearn.ensemble import GradientBoostingRegressor  # Modelo de regresión de Gradient Boosting

# Importación de funciones personalizadas
from utils import update_model, save_simple_metrics_report, get_model_performance_test_set  # Funciones para actualizar modelo, guardar reporte y graficar rendimiento
import logging  # Módulo para manejo de registros de eventos
import sys  # Módulo para acceso al sistema y flujo de salida
import numpy as np  # Biblioteca para operaciones numéricas eficientes
import pandas as pd  # Biblioteca para manipulación y análisis de datos

# Configuración básica del registro de eventos (logging)
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',  # Formato del mensaje de registro
    level=logging.INFO,  # Nivel de registro: INFO (mostrar mensajes informativos)
    datefmt='%H:%M:%S',  # Formato del tiempo en el registro
    stream=sys.stderr  # Flujo de salida al que se enviarán los registros (consola)
)

logger = logging.getLogger(__name__)  # Instancia de logger específica para este script

# Mensaje informativo de carga de datos
logger.info('Loading Data...')
data = pd.read_csv('dataset/full_data.csv')  # Cargar datos desde un archivo CSV

# Mensaje informativo de carga del modelo
logger.info('Loading model')
# Definición de un pipeline de procesamiento y modelo
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),  # Imputación de valores faltantes con la media
    ('core_model', GradientBoostingRegressor())  # Modelo de regresión de Gradient Boosting
])

# Mensaje informativo de separación de datos en entrenamiento y prueba
logger.info('Separating dataset into train and test')
X = data.drop('worldwide_gross', axis=1)  # Features: todas las columnas excepto 'worldwide_gross'
y = data['worldwide_gross']  # Target: columna 'worldwide_gross'

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Mensaje informativo sobre la configuración de los hiperparámetros para ajustar
logger.info('Setting Hyperparameter to tune')
param_tuning = {'core_model__n_estimators': range(20, 301, 30)}  # Diccionario de parámetros para ajustar

# Configuración de la búsqueda en malla (grid search)
grid_search = GridSearchCV(model, param_grid=param_tuning, scoring='r2', cv=5)

# Mensaje informativo sobre el inicio de la búsqueda en malla
logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)  # Ejecución de la búsqueda en malla con los datos de entrenamiento

# Mensaje informativo sobre la validación cruzada con el mejor modelo encontrado
logger.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5)

# Cálculo de las puntuaciones promedio de entrenamiento y prueba
train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])

# Asegurándose de que las puntuaciones sean adecuadas según criterios definidos
assert train_score > 0.7
assert test_score > 0.65

# Registro de las puntuaciones de entrenamiento y prueba
logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

# Mensaje informativo sobre la actualización del modelo con el mejor estimador encontrado
logger.info('Updating model')
update_model(grid_search.best_estimator_)  # Actualización del modelo guardando el mejor estimador

# Mensaje informativo sobre la generación del informe del modelo
logger.info('Generating model report...')
validation_score = grid_search.best_estimator_.score(X_test, y_test)  # Evaluación del modelo en el conjunto de prueba
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)  # Guardar un reporte con métricas del modelo

# Generación y guardado del gráfico de rendimiento del modelo en el conjunto de prueba
y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

# Mensaje informativo de finalización del entrenamiento
logger.info('Training Finished')
