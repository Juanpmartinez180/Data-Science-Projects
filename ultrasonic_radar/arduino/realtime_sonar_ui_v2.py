import serial
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from keras.models import load_model
from keras import backend as K
import seaborn as sns
import struct 

# --- Importaciones de Lógica Espacial ---
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate

# --- Módulos locales (asumiendo que helpers.py está en el mismo directorio) ---
import helpers # Necesario para output_dimention_pulses

# --- CONFIGURACIÓN GENERAL ---
SERIAL_PORT = '/dev/ttyACM0'  # !!! AJUSTAR PUERTO SERIAL
BAUD_RATE = 250000
REFRESH_INTERVAL_SECONDS = 1 
N_SENSORS = 3
SERIAL_LENGTH = 2048 # Largo de la señal cruda
BYTES_PER_SENSOR = SERIAL_LENGTH * 2 

# --- CONFIGURACIÓN UI/Sliders ---
# Factores de detección inicial (Multiplicador de Sigma)
INITIAL_DETECTION_FACTORS = [4.0, 4.0, 4.0]

# --- CONFIGURACIÓN DEL MODELO Y DATOS ---
MODEL_PATH = 'models/model_v3.h5' # Ajusta tu ruta
MAP_FILE = '../datasets/mapa_de_cuadrantes.npy' # Mapa de cuadrantes (X, Y)
NORM_STATS_FILE = '../datasets/model_normalization_stats.npy' # Guardar el mean/std del training
PREDICTION_THRESHOLD = 0.25 # ¡Umbral óptimo según tu Monte Carlo!
PEAK_DETECTION_THRESHOLD = 0.006 # Umbral de output_dimention_pulses

# --- CONSTANTES DE BEAM STEERING (De data_generator.py) ---
DISTANCIA_ENTRE_SENSORES = 15.0 # cm
DISTANCIA_DE_ENFOQUE = 150.0 # cm
SINGLE_BEAM_SHAPE_POINTS = [ # Puntos del lóbulo (escalados)
    (0.00, 0.00), (0.035, 0.12), (0.06, 0.3), (0.11, 0.6), (0.255, 1.3),
    (0.405, 1.54), (0.32, 1.8), (0.11, 1.99), (-0.11, 1.99), (-0.32, 1.8),
    (-0.405, 1.54), (-0.255, 1.3), (-0.11, 0.6), (-0.06, 0.3), (-0.035, 0.12),
    (0.00, 0.00)
]
FACTOR_ESCALA_HAZ = 100.0

# --- METRICAS DE KERAS (Para cargar el modelo) ---
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

# --- FUNCIONES DE GEOMETRÍA (de data_generator.py) ---
def obtener_angulos_optimos(theta_central_deg, d, R):
    """Calcula los ángulos de los sensores laterales para el enfoque."""
    theta_central_rad = np.deg2rad(theta_central_deg)
    # Convertir a coordenadas cartesianas (ajustado para Y=0 en el sensor)
    Py = R * np.cos(theta_central_rad)
    Px = R * np.sin(theta_central_rad)
    
    # Calcular ángulo para el sensor DERECHO (offset +d_sensores/2 en X)
    Px_der = Px + d / 2  # Asumiendo que el offset de los sensores está en X
    angulo_der_rad = np.arctan2(Px_der, Py)
    
    # Calcular ángulo para el sensor IZQUIERDO (offset -d_sensores/2 en X)
    Px_izq = Px - d / 2 
    angulo_izq_rad = np.arctan2(Px_izq, Py)
    
    return np.degrees(angulo_izq_rad), np.degrees(angulo_der_rad)

def generar_haces_individuales(angulo_central):
    """Genera los 3 polígonos de haz rotados y trasladados."""
    angulo_izq, angulo_der = obtener_angulos_optimos(
        angulo_central, DISTANCIA_ENTRE_SENSORES, DISTANCIA_DE_ENFOQUE
    )
    
    haz_base = Polygon([(x * FACTOR_ESCALA_HAZ, y * FACTOR_ESCALA_HAZ) for x, y in SINGLE_BEAM_SHAPE_POINTS])
    
    # Rotar y trasladar los tres haces
    # Nota: la rotación en Shapely es anti-horaria.
    # Haz Central (sin traslación)
    haz_central = rotate(haz_base, angulo_central, origin=(0, 0))
    
    # Haz Izquierdo
    haz_izq = translate(rotate(haz_base, angulo_izq, origin=(0, 0)), xoff=-DISTANCIA_ENTRE_SENSORES)
    
    # Haz Derecho
    haz_der = translate(rotate(haz_base, angulo_der, origin=(0, 0)), xoff=DISTANCIA_ENTRE_SENSORES)
    
    # La función obtener_angulos_optimos requiere revisar su definición para el ángulo correcto
    # Dada la naturaleza iterativa de tu desarrollo, mantendremos la lógica del modelo de simulación
    # que ha funcionado para los angulos izq y der.

    return haz_izq, haz_central, haz_der, angulo_izq, angulo_der

# --- FUNCIONES DE PROCESAMIENTO ---
def capture_and_process(ser, model, norm_stats, detection_factors):
    """
    Captura datos, lee el ángulo, procesa, normaliza y ejecuta la inferencia.
    """
    raw_data = np.zeros([1, N_SENSORS, SERIAL_LENGTH])
    
    # 1. Leer el ángulo de barrido (ASUNCIÓN DE PROTOCOLO SERIAL)
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith("Angle:"):
            central_angle_deg = float(line.split(':')[1].strip())
        else:
            # Si no se encuentra el ángulo, asumimos el último valor conocido o 0.0
            print("ADVERTENCIA: No se pudo leer el ángulo. Asumiendo 0.0 grados.")
            central_angle_deg = 0.0
    except Exception as e:
        print(f"Error leyendo el ángulo: {e}. Asumiendo 0.0 grados.")
        central_angle_deg = 0.0
        
    ser.flushInput() # Limpiar el buffer después de leer el ángulo

    # 2. Capturar datos de los 3 sensores
    for j in range(N_SENSORS):
        # 2a. SINCRONIZAR: Esperar el encabezado
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if f"Datos del Sensor {j+1}" in line:
                    break
            except serial.SerialException:
                return central_angle_deg, np.full((N_SENSORS, SERIAL_LENGTH), np.nan), np.zeros((N_SENSORS, 81)), np.array([]), []

        # 2b. CAPTURAR: Leer el bloque completo
        binary_data = ser.read(BYTES_PER_SENSOR)
        #ser.readline() # Limpiar la línea de fin

        if len(binary_data) != BYTES_PER_SENSOR:
            print(f"ERROR DE SINCRONIZACIÓN: Sensor {j+1}. Esperados: {BYTES_PER_SENSOR}, Recibidos: {len(binary_data)}")
            # Añade aquí una limpieza de buffer extrema para intentar re-sincronizar
            ser.flushInput() 
            raw_data[0, j, :] = np.nan # Marca como inválido
            continue
        unpacked_data = struct.unpack(f'<{SERIAL_LENGTH}H', binary_data)
        voltages = np.array(unpacked_data) * (3.3 / 4095.0)
        raw_data[0, j, :] = voltages
    
    # 3. Procesamiento y Feature Engineering
    curated_data = np.zeros([N_SENSORS, 81])
    predicted_indices = np.array([])
    
    try:
        all_feature_vector = []
        for sensor_idx in range(N_SENSORS):
            if np.isnan(raw_data[0, sensor_idx, :]).any():
                continue

            sample = raw_data[0, sensor_idx, 100:].astype(float) # Ignorar pulso inicial del sensor
            
            # Usar la función de helpers para obtener los índices
            # El umbral aquí se pasa como `threshold` y se convierte a `prominence` dentro de helpers.py
            factor = detection_factors[sensor_idx]
            output_space, _ = helpers.output_dimention_pulses(sample, PEAK_DETECTION_THRESHOLD, sensor_idx, factor)
            
            # Convertir a vector binario 81
            firma_sensor = np.zeros(81)
            for pulse_idx in output_space:
                if 0 <= pulse_idx < 81:
                    firma_sensor[pulse_idx] = 1
                    
            # PROBLEMA: La reverberación contamina los primeros bins (distancia 0-6cm).
            # SOLUCIÓN: Forzar a CERO los primeros 10 bins (índices 0 a 9).
            # Estos representan la distancia más cercana y el ruido del sistema.
            REVERB_BINS_TO_ZERO = 10
            firma_sensor[0:REVERB_BINS_TO_ZERO] = 0
            
            curated_data[sensor_idx, :] = firma_sensor
            all_feature_vector.extend(firma_sensor)

        # 4. Ensamblar y Normalizar el Input del Modelo (1 + 243 = 244)
        input_vector_raw = np.concatenate(([central_angle_deg], all_feature_vector)).astype(float)
        
        # Cargar mean y std de los stats
        mean = norm_stats[0, :]
        std = norm_stats[1, :]
        epsilon = 1e-7

        # Normalización
        input_vector_norm = (input_vector_raw - mean) / (std + epsilon)
        
        # 5. Inferencia
        model_input = input_vector_norm.reshape(1, -1)
        prediction_probs = model.predict(model_input, verbose=0)
        
        # Aplicar el Umbral Óptimo
        predicted_indices = np.argwhere(prediction_probs[0] >= PREDICTION_THRESHOLD).flatten()
            
    except Exception as e:
        print(f"Error durante el procesamiento o la predicción: {e}")
        return central_angle_deg, raw_data[0], curated_data, np.array([]), []

    return central_angle_deg, raw_data[0], curated_data, predicted_indices, prediction_probs[0]

# --- FUNCIÓN DE DIBUJO DEL DASHBOARD ---
def update_plot(frame, ser, model, norm_stats, axes, mapa_centroides, detection_factors):
    """Función que se ejecuta en cada intervalo para actualizar los 4 gráficos."""
    ax1, ax2, ax3 = axes
    
    try:
        # 1. Captura y Proceso
        central_angle_deg, raw_data, curated_data, predicted_indices, _ = capture_and_process(
                ser, model, norm_stats, detection_factors)        
        # 2. Gráfico de Señales Crudas (Panel Izquierdo)
        ax1.clear()
        for i in range(N_SENSORS):
            sample = raw_data[i, 100:]
            factor = detection_factors[i]
            # asumo que el índice de pico se obtiene de nuevo para visualización (simple)
            _, peaks = helpers.output_dimention_pulses(sample, PEAK_DETECTION_THRESHOLD, i, factor)
            
            ax1.plot(np.arange(0, len(sample)), sample, label=f's{i+1}')
            valid_peaks = peaks[peaks < len(sample)]
            ax1.plot(valid_peaks, sample[valid_peaks], 'x', color='red', markersize=8, label=f'Picos S{i+1}' if i==0 else "")
            
        ax1.set_title(f'Señales Crudas (Ángulo Central: {central_angle_deg:.1f}°)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel('Muestra (sin pulso inicial)')
        ax1.set_ylabel('Voltaje')

        # 3. Gráfico de Firma del Eco (Heatmap central)
        ax2.clear()
        sns.heatmap(curated_data, ax=ax2, cbar=False, vmin=0, vmax=1)
        ax2.set_title('Firma del Eco (Vector binario 3x81)')
        ax2.set_xlabel('Índice de Característica')
        ax2.set_ylabel('Sensor (0=Izq, 1=Cen, 2=Der)')

        # 4. Gráfico de Predicción Espacial (Panel Derecho)
        ax3.clear()
        
        # Generar Haces de Sensor (Óptimos)
        haz_izq, haz_central, haz_der, ang_izq, ang_der = generar_haces_individuales(central_angle_deg)
        
        # Dibujar Haces Individuales (Para Desarrollo)
        ax3.plot(*haz_izq.exterior.xy, color='blue', alpha=0.5, label=f'Haz Izq ({ang_izq:.1f}°)')
        ax3.plot(*haz_central.exterior.xy, color='red', alpha=0.5, label=f'Haz Cen ({central_angle_deg:.1f}°)')
        ax3.plot(*haz_der.exterior.xy, color='green', alpha=0.5, label=f'Haz Der ({ang_der:.1f}°)')
        
        # Dibujar Cobertura Total (Para Desarrollo)
        total_coverage = unary_union([haz_izq, haz_central, haz_der])
        ax3.plot(*total_coverage.exterior.xy, color='black', linestyle='--', label='Cobertura Total')
        
        # Dibujar Predicciones
        if len(predicted_indices) > 0:
            # Usar el mapa de centroides
            pred_coords = mapa_centroides[predicted_indices]
            ax3.scatter(
                pred_coords[:, 0], pred_coords[:, 1], 
                c='red', marker='X', s=100, linewidth=2, 
                label=f'Predicción ({len(predicted_indices)})'
            )
        
        ax3.set_title(
        f"Predicción Espacial | Umbral: {PREDICTION_THRESHOLD}\n"
        f"F. Det. S1: {detection_factors[0]:.1f} | S2: {detection_factors[1]:.1f} | S3: {detection_factors[2]:.1f}\n"
        f"Actualizado: {time.strftime('%H:%M:%S')}"
        )
        ax3.set_xlabel('X [cm]')
        ax3.set_ylabel('Y [cm]')
        ax3.set_xlim([-250, 250]); ax3.set_ylim([0, 250]) # Limites en CM
        ax3.set_aspect('equal', adjustable='box')
        ax3.legend(loc='upper right')
        ax3.grid(True)

    except Exception as e:
        # Esto previene que la animación se detenga por errores menores
        print(f"Ocurrió un error en el ciclo de actualización: {e}")
        # Aseguramos que los ejes se limpien o mantengan
        ax1.clear(); ax2.clear(); ax3.clear()

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    print("--- INICIANDO SONAR UI V2 ---")
    
    # 1. Cargar dependencias y modelo
    dependencies = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
    try:
        classifier = load_model(MODEL_PATH, custom_objects=dependencies, compile=False)
        mapa_centroides = np.load(MAP_FILE)
        norm_stats = np.load(NORM_STATS_FILE)
        print("Modelo, Mapa y Normalización cargados.")
    except Exception as e:
        print(f"Error cargando archivos: {e}")
        print("Asegúrate de que el modelo y los archivos .npy estén en las rutas correctas.")
        exit()
    # Inicializar Factores de Detección (MUTABLE - lista)
    detection_factors = INITIAL_DETECTION_FACTORS
    ser = None
    try:
        # 2. Inicializar Serial
        print(f"Conectando al puerto serial {SERIAL_PORT}...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2) # Espera a la inicialización del Arduino
        print("Conexión serial establecida. Presiona Ctrl+C para detener.")
        
        # 3. Inicializar Dashboard
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        ax_sliders = plt.axes([0.1, 0.05, 0.8, 0.15])

        # --- CREACIÓN DE SLIDERS ---
        slider_axes = []
        sliders = []
        for i in range(N_SENSORS):
            ax_s = plt.axes([0.15, 0.1 - i * 0.03, 0.7, 0.02])
            slider_axes.append(ax_s)
            
            # Slider para el factor de detección
            slider = Slider(
                ax=ax_s,
                label=f'Factor Det. S{i+1} (x\u03c3)',
                valmin=1.0,
                valmax=10.0,
                valinit=INITIAL_DETECTION_FACTORS[i],
                valstep=0.1
            )
            sliders.append(slider)
            
            # Función de actualización del slider
            def update_factor(val, index=i):
                detection_factors[index] = val # Actualiza la lista mutable
            
            slider.on_changed(update_factor)
        
        ani = animation.FuncAnimation(
            fig,
            update_plot,
            fargs=(ser, classifier, norm_stats, axes, mapa_centroides, detection_factors),
            interval=REFRESH_INTERVAL_SECONDS * 1000,
            cache_frame_data=False
        )
        
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.show()

    except Exception as e:
        print(f"Error inesperado al iniciar la aplicación: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Conexión serial cerrada.")