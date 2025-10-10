import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import load_model
from keras import backend as K
import seaborn as sns
import struct 
# Nuevas librerías necesarias para el gráfico espacial (de object_location.py)
import shapely
from shapely.geometry import Polygon, Point
from shapely import affinity
import geopandas as gpd

# --- Módulos locales ---
import helpers
import object_location # Aún lo usamos para la función make_grid

# --- CONFIGURACIÓN ---
SERIAL_PORT = '/dev/ttyACM0'  # ¡IMPORTANTE! Cambia esto al puerto serial de tu Arduino
BAUD_RATE = 250000
MODEL_PATH = '../models/model_v2.h5' # Ruta al modelo
REFRESH_INTERVAL_SECONDS = 1 # Tiempo de refresco
PREDICTION_THRESHOLD = 0.1
PEAK_DETECTION_THRESHOLD = 0.1 # Umbral para la nueva detección de picos
N_SENSORS = 3
SERIAL_LENGTH = 2048

# --- MÉTRICAS PERSONALIZADAS DE KERAS ---
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# --- FUNCIONES DE PROCESAMIENTO ---
def capture_and_process(ser, model):
    """
    Captura datos binarios, los procesa y ejecuta la inferencia del modelo.
    """
    # Array para almacenar los voltajes finales
    raw_data = np.zeros([1, N_SENSORS, SERIAL_LENGTH])
    
    # Constante para la lectura binaria: 2048 muestras * 2 bytes/muestra (uint16_t)
    BYTES_PER_SENSOR = SERIAL_LENGTH * 2 
    
    ser.flushInput()

    for j in range(N_SENSORS):
        # 1. SINCRONIZAR: Esperar el encabezado (esto no cambia)
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if f"Datos del Sensor {j+1}" in line:
                    print(f"Recibiendo datos binarios del sensor {j+1}...")
                    break
            except serial.SerialException as e:
                print(f"Error de comunicación serial: {e}")
                # Devolvemos arrays vacíos o con NaNs para indicar el fallo
                return np.full((N_SENSORS, SERIAL_LENGTH), np.nan), np.zeros((N_SENSORS, 81)), np.array([]), []

        # 2. CAPTURAR: Leer el bloque completo de bytes binarios
        binary_data = ser.read(BYTES_PER_SENSOR)
        
        # 3. Limpiar: Leer la línea de fin de datos para limpiar el buffer
        ser.readline() 

        # 4. PROCESAR: Decodificar los bytes y convertirlos a voltaje
        if len(binary_data) == BYTES_PER_SENSOR:
            # '<' indica little-endian (correcto para Arduino)
            # '{SERIAL_LENGTH}H' indica 2048 enteros cortos sin signo
            unpacked_data = struct.unpack(f'<{SERIAL_LENGTH}H', binary_data)
            
            # Convertir la tupla de enteros (0-4095) a un array de voltajes
            voltages = np.array(unpacked_data) * (3.3 / 4095.0)
            
            raw_data[0, j, :] = voltages
        else:
            print(f"Error de lectura para el sensor {j+1}: Se recibieron {len(binary_data)} bytes en lugar de {BYTES_PER_SENSOR}.")
            # Rellenar con NaNs (Not a Number) para indicar datos corruptos
            raw_data[0, j, :] = np.nan

    # --- El resto del procesamiento y la inferencia no necesitan cambios ---
    
    curated_data = np.zeros([1, N_SENSORS, 81])
    predicted_indices = np.array([])
    all_peaks = []
    
    try:
        for sensor_idx in range(raw_data.shape[1]):
            # Asegurarse de que no hay NaNs antes de procesar
            if np.isnan(raw_data[0, sensor_idx, :]).any():
                print(f"Omitiendo procesamiento para el sensor {sensor_idx+1} debido a datos corruptos.")
                all_peaks.append([]) # Agregar una lista vacía para los picos
                continue

            sample = raw_data[0, sensor_idx, 100:].astype(float)
            
            # Asumo que 'helpers' está importado en tu script principal
            output_space, peaks = helpers.output_dimention_pulses(sample, PEAK_DETECTION_THRESHOLD)
            all_peaks.append(peaks) # Guarda los picos para graficarlos
            
            print(f"Sensor {sensor_idx+1} - Índices de picos detectados: {output_space}")
            for pulse_idx in output_space:
                if 0 <= pulse_idx < curated_data.shape[2]:
                    curated_data[0, sensor_idx, int(pulse_idx)] = 1

        if np.any(curated_data):
            model_input = curated_data.reshape(1, -1)
            prediction_probs = model.predict(model_input, verbose=0)
            predicted_indices = np.argwhere(prediction_probs[0] >= PREDICTION_THRESHOLD).flatten()
            print(f"Cuadrantes predichos (índices): {predicted_indices}")
        else:
            print("No se detectaron ecos claros en ningún sensor.")
            
    except Exception as e:
        print(f"Error durante el procesamiento de picos o la predicción: {e}")
        return raw_data[0], curated_data[0], np.array([]), all_peaks # Devuelve lo que se tenga hasta el momento

    return raw_data[0], curated_data[0], predicted_indices, all_peaks

# --- FUNCIÓN DE DIBUJO DEL DASHBOARD ---
def update_plot(frame, ser, model, axes, coords_df):
    """Función que se ejecuta en cada intervalo para actualizar los 3 gráficos."""
    ax1, ax2, ax3 = axes
    try:
        raw_data, curated_data, predicted_indices, all_peaks = capture_and_process(ser, model)
        
        # Gráfico 1: Datos Crudos
        ax1.clear()
        for i in range(N_SENSORS):
            sample = raw_data[i, 100:]
            ax1.plot(np.arange(0, len(sample)), sample, label=f's{i+1}')
            peaks = all_peaks[i]
            valid_peaks = peaks[peaks < len(sample)]
            ax1.plot(valid_peaks, sample[valid_peaks], 'x', color='red', markersize=8, label=f'Picos S{i+1}' if i==0 else "")
        ax1.set_title('Señales Crudas del Sensor')
        ax1.legend()
        ax1.grid(True)

        # Gráfico 2: Firma del Eco
        ax2.clear()
        sns.heatmap(curated_data, ax=ax2, cbar=False, vmin=0, vmax=1)
        ax2.set_title('Firma del Eco')
        ax2.set_xlabel('Índice')
        ax2.set_ylabel('Sensor')

        # Gráfico 3: Predicción Espacial (Lógica de object_location.test)
        ax3.clear()
        # 1. Crear la geometría del haz (cono)
        p1 = Point(0,0); p2 = Point(0.26, 1.983); p3 = Point(-0.26, 1.983)
        triangle1 = Polygon([p1, p2, p3])
        circle1 = Point(0, 1).buffer(1, resolution=50000)
        geometry_1 = triangle1.intersection(circle1)
        geometry_2 = affinity.rotate(geometry_1, -5, (-0.26, 1.983))
        geometry_3 = affinity.rotate(geometry_1, 5, (0.26, 1.983))
        int_1 = geometry_1.intersection(geometry_2)
        int_2 = geometry_1.intersection(geometry_3)
        beam_area = int_1.union(int_2)
        grid_geometry = object_location.make_grid(beam_area, 0.06)

        # 2. Dibujar el área del haz
        x, y = beam_area.exterior.xy
        ax3.plot(x, y, label='Beam Area', color='black')
        
        # 3. Dibujar las predicciones
        # Para que la leyenda no se repita, la creamos una sola vez
        predicted_label_added = False
        for object_idx in predicted_indices:
            coords = np.dstack(grid_geometry[object_idx].centroid.coords.xy).tolist()[0][0]
            temp_obj = Point(coords)
            x, y = temp_obj.buffer(0.021).exterior.xy
            
            label = ''
            if not predicted_label_added:
                label = 'Predicted object'
                predicted_label_added = True
            
            ax3.plot(x, y, color='blue', label=label)

        if not predicted_label_added:
             ax3.text(0, 1, 'No se detectó objeto', ha='center', va='center', fontsize=12, color='gray')
            
        ax3.set_title(f"Predicción Espacial\nActualizado: {time.strftime('%H:%M:%S')}")
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_xlim([-1, 1]); ax3.set_ylim([0, 2])
        ax3.set_aspect('equal', adjustable='box')
        ax3.legend(loc='upper right')

    except Exception as e:
        print(f"Ocurrió un error en el ciclo de actualización: {e}")

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    print("Cargando modelo de Keras...")
    dependencies = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
    classifier = load_model(MODEL_PATH, custom_objects=dependencies, compile=False)
    print("Modelo cargado exitosamente.")
    
    coords = pd.read_csv('../datasets/coords.csv')

    ser = None
    try:
        print(f"Conectando al puerto serial {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        print("Conexión serial establecida.")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        ani = animation.FuncAnimation(
            fig,
            update_plot,
            fargs=(ser, classifier, axes, coords),
            interval=REFRESH_INTERVAL_SECONDS * 1000,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error inesperado al iniciar la aplicación: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Conexión serial cerrada.")