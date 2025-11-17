import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # <-- Ya no necesitamos TextBox
from keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import os

# --- FASE 1: Cargar Métricas Personalizadas y Modelo ---
# (Necesario para que Keras pueda cargar el modelo)

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

# Nombres de los archivos
MODEL_FILE = 'sonar_model_multiobjeto.h5'
X_FILE = 'datasets/features_150k_17_11_25.csv.npy'
Y_FILE = 'datasets/labels_150k_17_11_25.csv.npy'
MAP_FILE = 'mapa_de_cuadrantes.npy'

# --- FASE 2: Cargar Modelo y Datos ---

print("Cargando modelo y métricas personalizadas...")
try:
    custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
    model = load_model(MODEL_FILE, custom_objects=custom_objects)
    print(f"Modelo '{MODEL_FILE}' cargado.")
except IOError:
    print(f"Error: No se pudo encontrar el archivo del modelo '{MODEL_FILE}'.")
    exit()

print("Cargando y preparando datos (esto puede tardar un momento)...")
try:
    # Cargar los datos COMPLETOS (necesario para replicar la normalización)
    X_full = np.load(X_FILE)
    y_full = np.load(Y_FILE)
    mapa_centroides = np.load(MAP_FILE)
except IOError as e:
    print(f"Error: No se pudo encontrar un archivo de datos. {e}")
    exit()

# 1. Replicar la división del notebook
X_train, X_test_raw, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# 2. Replicar la normalización del notebook
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
epsilon = 1e-7
X_test = (X_test_raw - mean) / (std + epsilon)

# 3. Replicar la conversión de tipo de 'y'
y_test = y_test.astype('float32')

# 4. Preparar el fondo del mapa (todos los cuadrantes)
all_x = mapa_centroides[:, 0]
all_y = mapa_centroides[:, 1]

print(f"Datos listos. {len(X_test)} muestras de prueba disponibles (0 a {len(X_test) - 1}).")

# --- FASE 3: Lógica de Visualización Interactiva ---

class PredictionVisualizer:
    def __init__(self):
        self.current_sample_index = 0
        self.current_threshold = 0.3 # Empezamos con el umbral que descubriste
        self.current_pred_probs = None # Caché para la predicción

        # Crear la figura y los ejes
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        plt.subplots_adjust(left=0.1, bottom=0.25) # Espacio para widgets
        
        # Ejes para los widgets
        ax_thresh = plt.axes([0.2, 0.15, 0.65, 0.03]) # [izq, abajo, ancho, alto]
        ax_sample = plt.axes([0.2, 0.1, 0.65, 0.03])  # Eje para el slider de muestra

        # Crear widgets
        self.thresh_slider = Slider(
            ax=ax_thresh,
            label='Umbral',
            valmin=0.01,
            valmax=1.0,
            valinit=self.current_threshold,
            valstep=0.01
        )
        
        # --- NUEVO SLIDER PARA MUESTRAS ---
        self.sample_slider = Slider(
            ax=ax_sample,
            label='Muestra #',
            valmin=0,
            valmax=len(X_test) - 1, # Límite superior es el total de muestras de test
            valinit=self.current_sample_index,
            valstep=1 # Importante: el paso es de 1 en 1
        )

        # Conectar eventos
        self.thresh_slider.on_changed(self.on_threshold_change)
        self.sample_slider.on_changed(self.on_sample_change) # Conectar al nuevo handler

        # Predicción inicial
        self.update_prediction()
        # Dibujo inicial
        self.draw_plot()

    def update_prediction(self):
        """Esta es la parte 'lenta', recalcula la predicción del modelo."""
        print(f"\nCalculando predicción para muestra {self.current_sample_index}...")
        sample_data = X_test[self.current_sample_index].reshape(1, -1)
        self.current_pred_probs = model.predict(sample_data)[0]
        print("Cálculo listo.")
    
    # --- NUEVO HANDLER PARA EL SLIDER DE MUESTRAS ---
    def on_sample_change(self, val):
        """Se llama al mover el slider de la muestra."""
        new_index = int(val) # El slider devuelve float, convertir a int
        
        # Solo recalcular si el índice realmente cambió
        if new_index != self.current_sample_index:
            self.current_sample_index = new_index
            self.update_prediction() # Recalcular (lento)
            self.draw_plot() # Redibujar (rápido)

    def on_threshold_change(self, val):
        """Se llama al mover el slider de umbral (rápido)."""
        self.current_threshold = val
        self.draw_plot() # Solo redibujar, no recalcular

    def draw_plot(self):
        """Esta es la parte 'rápida', solo redibuja el gráfico."""
        self.ax.clear()

        # 1. Obtener vectores
        y_true_vector = y_test[self.current_sample_index]
        y_pred_binary = (self.current_pred_probs > self.current_threshold).astype(int)

        # 2. Encontrar índices
        true_indices = np.where(y_true_vector == 1)[0]
        pred_indices = np.where(y_pred_binary == 1)[0]
        
        # 3. Encontrar coordenadas
        true_coords = mapa_centroides[true_indices]
        pred_coords = mapa_centroides[pred_indices]
        
        # 4. Calcular métricas en vivo para esta muestra
        tp = np.sum((y_true_vector == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_vector == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_vector == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 5. Dibujar
        self.ax.plot(all_x, all_y, 'o', color='gray', markersize=2, alpha=0.1, label='Grilla Completa')
        
        if len(true_coords) > 0:
            self.ax.plot(true_coords[:, 0], true_coords[:, 1], 'o', color='green', markersize=12, alpha=0.7, fillstyle='none', mew=2, label=f'Reales ({len(true_coords)})')
        
        if len(pred_coords) > 0:
            self.ax.plot(pred_coords[:, 0], pred_coords[:, 1], 'x', color='red', markersize=10, mew=2, label=f'Predichos ({len(pred_coords)})')
        
        title = (
            f"Muestra: {self.current_sample_index} | Umbral: {self.current_threshold:.2f}\n"
            f"F1: {f1:.3f} | Prec: {precision:.3f} | Recall: {recall:.3f}"
        )
        self.ax.set_title(title)
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.fig.canvas.draw_idle()

# --- FASE 4: Ejecutar el Visualizador ---
if __name__ == "__main__":
    # Verificar que todos los archivos necesarios existan
    files_needed = [MODEL_FILE, X_FILE, Y_FILE, MAP_FILE]
    if not all(os.path.exists(f) for f in files_needed):
        print("Error: Faltan uno o más archivos necesarios.")
        print(f"Asegúrate de que {', '.join(files_needed)} estén en el directorio correcto.")
    else:
        viz = PredictionVisualizer()
        print("\n--- Visualizador Interactivo Listo ---")
        print("Mueve los sliders para explorar.")
        plt.show()