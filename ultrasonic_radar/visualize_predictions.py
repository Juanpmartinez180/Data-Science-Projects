import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
import os

# --- CONFIGURACIÓN INICIAL ---
# Archivos
MODEL_FILE = 'sonar_model_multiobjeto.h5'
X_FILE = 'datasets/features_150k_17_11_25.csv.npy'
Y_FILE = 'datasets/labels_150k_17_11_25.csv.npy'
MAP_FILE = 'mapa_de_cuadrantes.npy'

# Métricas estándar (para cargar el modelo)
def recall_m(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())

def precision_m(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())

def f1_m(y_true, y_pred):
    p = precision_m(y_true, y_pred)
    r = recall_m(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# --- CARGA DE DATOS ---
print("Cargando modelo y datos...")
try:
    custom_objects = {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
    model = load_model(MODEL_FILE, custom_objects=custom_objects)
    X_full = np.load(X_FILE)
    y_full = np.load(Y_FILE)
    mapa_centroides = np.load(MAP_FILE)
except Exception as e:
    print(f"Error cargando archivos: {e}")
    exit()

# Preparación de datos (replicando notebook)
X_train, X_test_raw, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_test = (X_test_raw - mean) / (std + 1e-7)
y_test = y_test.astype('float32')
all_x = mapa_centroides[:, 0]
all_y = mapa_centroides[:, 1]

print(f"Listo. {len(X_test)} muestras de prueba.")

# --- LÓGICA DEL VISUALIZADOR ---

class SpatialVisualizer:
    def __init__(self):
        # Estado inicial
        self.sample_idx = 0
        self.threshold = 0.3
        self.tolerance = 12.0 # Radio de tolerancia inicial (cm)
        self.pred_probs = None

        # Configuración de la figura
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        plt.subplots_adjust(left=0.1, bottom=0.30) # Más espacio abajo para 3 sliders

        # Ejes para los widgets
        ax_sample = plt.axes([0.15, 0.05, 0.7, 0.03])
        ax_thresh = plt.axes([0.15, 0.10, 0.7, 0.03])
        ax_toler  = plt.axes([0.15, 0.15, 0.7, 0.03])

        # Widgets
        self.sl_sample = Slider(ax_sample, 'Muestra #', 0, len(X_test)-1, valinit=0, valstep=1)
        self.sl_thresh = Slider(ax_thresh, 'Umbral Pred.', 0.0, 1.0, valinit=self.threshold, valstep=0.01)
        self.sl_toler  = Slider(ax_toler, 'Radio Tol. (cm)', 0.0, 50.0, valinit=self.tolerance, valstep=1.0)

        # Conexiones
        self.sl_sample.on_changed(self.update_sample)
        self.sl_thresh.on_changed(self.update_params)
        self.sl_toler.on_changed(self.update_params)

        # Inicio
        self.predict()
        self.draw()

    def predict(self):
        """Recalcula la predicción de la red (lento)"""
        sample_data = X_test[self.sample_idx].reshape(1, -1)
        self.pred_probs = model.predict(sample_data, verbose=0)[0]

    def calculate_spatial_metrics(self, true_coords, pred_coords, radius):
        """
        Calcula métricas espaciales basadas en distancia.
        """
        spatial_tp = 0
        spatial_fp = 0
        spatial_fn = 0
        
        # Casos borde: sin objetos o sin predicciones
        if len(true_coords) == 0 and len(pred_coords) == 0:
            return 1.0, 1.0, 1.0, [] # Todo perfecto (vacío)
        if len(true_coords) > 0 and len(pred_coords) == 0:
            return 0.0, 0.0, 0.0, [] # Todo omitido

        # Matriz de distancias (filas=Reales, cols=Predichos)
        # Si no hay predicciones, cdist falla, manejado arriba
        dists = cdist(true_coords, pred_coords)
        
        # 1. SPATIAL RECALL (TP y FN)
        # Para cada objeto REAL, ¿tiene alguna predicción cerca?
        covered_true_indices = []
        for i in range(len(true_coords)):
            # Distancia mínima de este objeto real a cualquier predicción
            if len(pred_coords) > 0 and np.min(dists[i, :]) <= radius:
                spatial_tp += 1
                covered_true_indices.append(i)
            else:
                spatial_fn += 1
        
        # 2. SPATIAL PRECISION (FP)
        # Para cada PREDICCIÓN, ¿está cerca de algún objeto real?
        # Nota: Un objeto real puede validar múltiples predicciones cercanas 
        # (o podemos ser estrictos, aquí somos permisivos para empezar)
        if len(pred_coords) > 0:
            for j in range(len(pred_coords)):
                # Distancia mínima de esta predicción a cualquier objeto real
                if len(true_coords) > 0 and np.min(dists[:, j]) <= radius:
                    pass # Es válida (ya contada en TP o soporte de TP)
                else:
                    spatial_fp += 1 # Está lejos de todo
        
        # Ajuste de TP para precisión: 
        # Si tengo 1 objeto y 10 predicciones cerca, la precisión debería bajar.
        # Definición estricta: TP_prec = predicciones que tienen un objeto cerca.
        tp_for_precision = len(pred_coords) - spatial_fp

        # Cálculos finales
        prec = tp_for_precision / len(pred_coords) if len(pred_coords) > 0 else 0.0
        rec  = spatial_tp / len(true_coords) if len(true_coords) > 0 else 0.0
        f1   = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        return f1, prec, rec, covered_true_indices

    def update_sample(self, val):
        idx = int(val)
        if idx != self.sample_idx:
            self.sample_idx = idx
            self.predict()
            self.draw()

    def update_params(self, val):
        self.threshold = self.sl_thresh.val
        self.tolerance = self.sl_toler.val
        self.draw()

    def draw(self):
        self.ax.clear()
        
        # Obtener coordenadas
        y_true_idx = np.where(y_test[self.sample_idx] == 1)[0]
        y_pred_idx = np.where(self.pred_probs > self.threshold)[0]
        
        true_coords = mapa_centroides[y_true_idx]
        pred_coords = mapa_centroides[y_pred_idx]

        # Calcular Métricas Espaciales
        sp_f1, sp_prec, sp_rec, covered_idx = self.calculate_spatial_metrics(
            true_coords, pred_coords, self.tolerance
        )

        # --- DIBUJO ---
        # 1. Fondo
        self.ax.scatter(all_x, all_y, c='gray', s=2, alpha=0.1)

        # 2. Áreas de Tolerancia (Círculos alrededor de los reales)
        for i, coords in enumerate(true_coords):
            color = '#aaffaa' if i in covered_idx else '#ffaaaa' # Verde si detectado, Rojo si no
            circle = Circle((coords[0], coords[1]), self.tolerance, color=color, alpha=0.3)
            self.ax.add_patch(circle)
            # Centro real
            self.ax.plot(coords[0], coords[1], 'o', color='green', markersize=8, markeredgecolor='k')

        # 3. Predicciones
        if len(pred_coords) > 0:
            self.ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='blue', marker='x', s=100, linewidth=2, label='Predicción')

        # Configuración Ejes y Título
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-200, 200) # Ajusta según tu mapa
        self.ax.set_ylim(0, 250)    # Ajusta según tu mapa
        
        title_str = (f"Muestra {self.sample_idx} | F1 Espacial: {sp_f1:.2f}\n"
                     f"Prec. Espacial: {sp_prec:.2f} | Recall Espacial: {sp_rec:.2f}\n"
                     f"Tol: {self.tolerance}cm | Umbral: {self.threshold:.2f}")
        self.ax.set_title(title_str, fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right')

        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viz = SpatialVisualizer()
    plt.show()