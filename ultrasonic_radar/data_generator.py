# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import nearest_points, unary_union # <--- IMPORTADO unary_union
from shapely.affinity import rotate, translate
import random
import pandas as pd
from tqdm import tqdm
import multiprocessing

# ===================================================================
# FASE 1: Configuración y Parámetros
# ===================================================================
DISTANCIA_ENTRE_SENSORES = 15
DISTANCIA_DE_ENFOQUE = 150
RANGO_DE_ANGULO = 140.0
NUMERO_DE_PASOS = 9
ANGULOS_DE_BARRIDO = np.linspace(-RANGO_DE_ANGULO / 2, RANGO_DE_ANGULO / 2, NUMERO_DE_PASOS)
SINGLE_BEAM_SHAPE_POINTS = [
    (0.00, 0.00), (0.035, 0.12), (0.06, 0.3), (0.11, 0.6), (0.255, 1.3),
    (0.405, 1.54), (0.32, 1.8), (0.11, 1.99), (-0.11, 1.99), (-0.32, 1.8),
    (-0.405, 1.54), (-0.255, 1.3), (-0.11, 0.6), (-0.06, 0.3), (-0.035, 0.12),
    (0.00, 0.00)
]
FACTOR_ESCALA_HAZ = 100.0
MAX_OBJETOS_POR_ESCENA = 6
PROPORCIONES_OBJETOS = {'circulo': 0, 'rectangulo': 1, 'pared': 0}
RANGO_TAMAÑO_CIRCULO = [5.0, 12.0]; RANGO_TAMAÑO_RECTANGULO = [6.0, 60.0]; RANGO_TAMAÑO_PARED = [50.0, 100.0]
TAMAÑO_DEL_CUADRANTE = 6.0
VELOCIDAD_SONIDO = 34300; FRECUENCIA_MUESTREO_ADC = 140000; FRECUENCIA_MUESTREO_ML = 6800
NUM_MUESTRAS_TOTALES = 10

# ===================================================================
# FASE 2 y 3: Funciones Geométricas y de Generación de Escena
# ===================================================================
def obtener_angulos_optimos(theta_central_deg, d, R):
    theta_central_math_rad = np.deg2rad(theta_central_deg + 90)
    px = R * np.cos(theta_central_math_rad); py = R * np.sin(theta_central_math_rad)
    angulo_math_izq = np.arctan2(py, px + d); angulo_final_izq = np.rad2deg(angulo_math_izq) - 90
    angulo_math_der = np.arctan2(py, px - d); angulo_final_der = np.rad2deg(angulo_math_der) - 90
    return angulo_final_izq, angulo_final_der

def crear_haz_desde_puntos(shape_points, scale_factor):
    scaled_points = [(x * scale_factor, y * scale_factor) for x, y in shape_points]
    return Polygon(scaled_points)

def generar_haces_individuales(angulo_central):
    angulo_izq, angulo_der = obtener_angulos_optimos(angulo_central, DISTANCIA_ENTRE_SENSORES, DISTANCIA_DE_ENFOQUE)
    haz_base = crear_haz_desde_puntos(SINGLE_BEAM_SHAPE_POINTS, FACTOR_ESCALA_HAZ)
    haz_central = rotate(haz_base, angulo_central, origin=(0, 0))
    haz_izq = translate(rotate(haz_base, angulo_izq, origin=(0, 0)), xoff=-DISTANCIA_ENTRE_SENSORES)
    haz_der = translate(rotate(haz_base, angulo_der, origin=(0, 0)), xoff=DISTANCIA_ENTRE_SENSORES)
    return [haz_izq, haz_central, haz_der]

# ========= FUNCIÓN MODIFICADA =========
def generar_escena_aleatoria(area_de_generacion):
    escena = []; num_objetos = random.randint(1, MAX_OBJETOS_POR_ESCENA)
    # Ahora los límites se toman del área de los haces, que es más pequeña y específica
    min_x, min_y, max_x, max_y = area_de_generacion.bounds
    tipos = list(PROPORCIONES_OBJETOS.keys()); probabilidades = list(PROPORCIONES_OBJETOS.values())
    
    for _ in range(num_objetos):
        # Se intenta generar un objeto hasta 10 veces para asegurar que caiga dentro del área
        for i in range(10):
            px, py = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            
            # --> CAMBIO CLAVE: Se verifica que el punto esté DENTRO de la forma del haz
            if not area_de_generacion.contains(Point(px, py)):
                continue

            tipo_objeto = np.random.choice(tipos, p=probabilidades)
            objeto = None
            if tipo_objeto == 'circulo': objeto = Point(px, py).buffer(random.uniform(*RANGO_TAMAÑO_CIRCULO))
            elif tipo_objeto == 'rectangulo':
                lado = random.uniform(*RANGO_TAMAÑO_RECTANGULO)
                objeto = rotate(box(px - lado/2, py - lado/2, px + lado/2, py + lado/2), random.uniform(0, 360), origin='centroid')
            elif tipo_objeto == 'pared':
                longitud = random.uniform(*RANGO_TAMAÑO_PARED)
                objeto = rotate(LineString([(px - longitud/2, py), (px + longitud/2, py)]).buffer(2), random.uniform(0, 360), origin='centroid')
            
            if objeto and not any(obj_existente.contains(objeto.centroid) for obj_existente in escena): 
                escena.append(objeto)
                break # Si el objeto es válido y se añade, salimos del bucle de reintentos
    return escena

# ===================================================================
# FASE 4: Simulación de Ray-Tracing con Oclusión
# ===================================================================
def simular_reflexiones_con_oclusion(escena, haces_individuales):
    origenes = [Point(-DISTANCIA_ENTRE_SENSORES, 0), Point(0, 0), Point(DISTANCIA_ENTRE_SENSORES, 0)]
    ecos_por_sensor = []
    puntos_de_reflexion = [] 

    for haz, origen in zip(haces_individuales, origenes):
        objetos_en_haz = [obj for obj in escena if obj.intersects(haz)]
        if not objetos_en_haz:
            ecos_por_sensor.append([]); continue
        
        objetos_en_haz.sort(key=lambda obj: origen.distance(obj))
        ecos_sensor_actual = []
        area_ocluida = Polygon()
        
        for obj in objetos_en_haz:
            
            parte_en_haz = obj.intersection(haz)
            # Primero, vemos si la línea de visión directa al punto más cercano del objeto está bloqueada.
            _, punto_mas_cercano = nearest_points(origen, parte_en_haz)
            linea_de_vision = LineString([origen, punto_mas_cercano])
            linea_bloqueada = area_ocluida.intersects(linea_de_vision)
            # Luego, calculamos la parte visible restando los oclusores que se tocan directamente.
            parte_visible_real = parte_en_haz.difference(area_ocluida)
            
            if not linea_bloqueada and not parte_visible_real.is_empty:
                _, punto_eco_en_objeto = nearest_points(origen, parte_visible_real)
                distancia_eco = origen.distance(punto_eco_en_objeto)
                
                ecos_sensor_actual.append(distancia_eco)
                puntos_de_reflexion.append(punto_eco_en_objeto)
                
            area_ocluida = area_ocluida.union(obj)
        ecos_por_sensor.append(ecos_sensor_actual)
        
    return ecos_por_sensor, puntos_de_reflexion

# ===================================================================
# FASE 5: Generación del Dataset (Lógica Principal Actualizada)
# ===================================================================
def generar_label_desde_detecciones(puntos_de_reflexion, cuadrantes_globales):
    label = np.zeros(len(cuadrantes_globales))
    if not puntos_de_reflexion:
        return label

    cuadrantes_asignados = set()
    for punto in puntos_de_reflexion:
        for i, cuadrante in enumerate(cuadrantes_globales):
            if i not in cuadrantes_asignados and cuadrante.intersects(punto):
                label[i] = 1
                cuadrantes_asignados.add(i)
                break
    return label

def distancias_a_firma_eco(ecos_por_sensor):
    firma_eco_total = []; ratio_conversion = FRECUENCIA_MUESTREO_ML / FRECUENCIA_MUESTREO_ADC
    for distancias_cm in ecos_por_sensor:
        firma_sensor = np.zeros(81)
        for dist_cm in distancias_cm:
            tiempo_eco = (2 * dist_cm) / VELOCIDAD_SONIDO
            indice_adc = int(tiempo_eco * FRECUENCIA_MUESTREO_ADC)
            indice_ml = int((indice_adc - 100) * ratio_conversion)
            if 0 <= indice_ml < 81:
                firma_sensor[indice_ml] = 1
        firma_eco_total.extend(firma_sensor)
    return np.array(firma_eco_total)

def generar_una_muestra(cuadrantes_globales):
    while True:
            # Primero se define el ángulo y se generan los haces.
            angulo_central = random.choice(ANGULOS_DE_BARRIDO)
            haces = generar_haces_individuales(angulo_central)
            
            # --> CAMBIO DE LÓGICA 2: Se crea una geometría única que representa toda el área de los haces.
            area_de_haces = unary_union(haces)
            
            # --> CAMBIO DE LÓGICA 3: Se genera la escena DENTRO de esa área específica.
            escena = generar_escena_aleatoria(area_de_haces)
            if not escena: continue

            # 1. Simular ecos (Features) Y OBTENER LOS PUNTOS DE REFLEXIÓN
            ecos, puntos_de_reflexion = simular_reflexiones_con_oclusion(escena, haces)
            
            # Si no hay ecos, no es una muestra útil para el entrenamiento (en este nuevo enfoque).
            # Se podría cambiar si también se quieren muestras "vacías".
            if not puntos_de_reflexion:
                continue

            firma_eco = distancias_a_firma_eco(ecos)
            
            # 2. Generar la etiqueta (Label) BASADA EN LOS PUNTOS DE REFLEXIÓN
            label = generar_label_desde_detecciones(puntos_de_reflexion, cuadrantes_globales)
            
            # Solo se guardan muestras donde se detectó algo.
            if sum(firma_eco) > 0 and sum(label) > 0:
                feature = np.concatenate(([angulo_central], firma_eco))
                return (feature, label)

# --- Bucle Principal de Generación ---
if __name__ == "__main__":
    features_list, labels_list = [], []
    ##escena_list, haces_list, angulo_central_list, firma_eco_list = [],[],[],[]
    
    area_total_visible = box(-250, 0, 250, 220)
    x_coords = np.arange(area_total_visible.bounds[0], area_total_visible.bounds[2], TAMAÑO_DEL_CUADRANTE)
    y_coords = np.arange(area_total_visible.bounds[1], area_total_visible.bounds[3], TAMAÑO_DEL_CUADRANTE)
    cuadrantes_globales = [box(x, y, x + TAMAÑO_DEL_CUADRANTE, y + TAMAÑO_DEL_CUADRANTE) for x in x_coords for y in y_coords]

    num_cores = multiprocessing.cpu_count()
    print(f"Usando {num_cores} núcleos para generar {NUM_MUESTRAS_TOTALES} muestras con oclusión mejorada...")

    args_list = [cuadrantes_globales] * NUM_MUESTRAS_TOTALES
    
    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(generar_una_muestra, args_list), total=NUM_MUESTRAS_TOTALES))

    features_list, labels_list = zip(*results)
    print(f"\nGeneración completada: {len(features_list)} muestras consistentes creadas.")

    # --- Guardado ---
    df_features = pd.DataFrame(features_list)
    df_labels = pd.DataFrame(labels_list)
    df_features.to_csv("features_150k.csv", index=False, header=False)
    df_labels.to_csv("labels_150k.csv", index=False, header=False)
    print("Archivos CSV guardados.")