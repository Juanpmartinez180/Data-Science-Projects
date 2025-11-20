import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct 
from scipy.signal import hilbert
import sys
import os

# --- Módulos locales ---
# Asumimos que helpers.py está presente, aunque solo usamos la lógica de procesamiento.

# --- CONFIGURACIÓN SERIAL ---
SERIAL_PORT = '/dev/ttyACM0'  # !!! AJUSTAR PUERTO SERIAL
BAUD_RATE = 250000
N_SENSORS = 3
SERIAL_LENGTH = 2048
BYTES_PER_SENSOR = SERIAL_LENGTH * 2 

# --- CONFIGURACIÓN DE CALIBRACIÓN ---
NOISE_STATS_FILE = '../datasets/sensor_noise_stats.npy'
SAMPLES_TO_AVERAGE = 5  # Número de capturas para promediar el ruido base

# --- FUNCIONES DE PROCESAMIENTO MÍNIMO ---
def process_signal_for_noise(raw_voltages):
    """
    Procesa la señal para obtener la desviación estándar de la envolvente del ruido.
    """
    # 1. Quitar el pulso inicial del sensor (asumiendo que las primeras 100 muestras son la resonancia)
    sample = raw_voltages[100:].astype(float)
    
    # 2. Obtener la envolvente de la señal (Hilbert)
    analytic_signal = hilbert(sample)
    envelope = np.abs(analytic_signal)
    
    # 3. Centrar la envolvente alrededor de la línea base (mediana)
    baseline = np.median(envelope)
    envelope_zero_centered = envelope - baseline
    
    # 4. Calcular la desviación estándar del ruido después del pulso inicial.
    #    Usamos una sección segura, por ejemplo, muestras 600 a 1000, asumiendo que es ruido puro.
    #    Si el objeto más lejano está más cerca de 1000, ajusta el rango.
    NOISE_SECTION_START = 600
    NOISE_SECTION_END = 1000
    
    if len(envelope_zero_centered) < NOISE_SECTION_END:
        return np.nan # Datos insuficientes

    noise_section = envelope_zero_centered[NOISE_SECTION_START:NOISE_SECTION_END]
    
    # La estadística de ruido es la desviación estándar de la envolvente de ruido.
    return np.std(noise_section)

def capture_raw_data(ser):
    """Captura una tanda de datos sin procesamiento de ángulo."""
    raw_data = np.zeros([N_SENSORS, SERIAL_LENGTH])
    
    ser.flushInput()

    for j in range(N_SENSORS):
        # Sincronización: Esperar el encabezado
        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if f"Datos del Sensor {j+1}" in line:
                    break
            except Exception:
                return np.full((N_SENSORS, SERIAL_LENGTH), np.nan)

        # Capturar datos binarios
        binary_data = ser.read(BYTES_PER_SENSOR)
        ser.readline() # Limpiar línea de fin

        if len(binary_data) == BYTES_PER_SENSOR:
            unpacked_data = struct.unpack(f'<{SERIAL_LENGTH}H', binary_data)
            voltages = np.array(unpacked_data) * (3.3 / 4095.0)
            raw_data[j, :] = voltages
        else:
            raw_data[j, :] = np.nan
            
    return raw_data

def update_plot(frame, ser, ax, noise_buffer, fig): # <<--- ACEPTA 'fig'
    raw_data = capture_raw_data(ser)
    
    ax.clear()
    
    for i in range(N_SENSORS):
        if not np.isnan(raw_data[i, :]).any():
            sigma = process_signal_for_noise(raw_data[i, :])
            if not np.isnan(sigma):
                noise_buffer[i].append(sigma)
            
            # Dibujar la señal sin el pulso inicial
            ax.plot(raw_data[i, 100:], label=f's{i+1}')
        
        # Mostrar la media del ruido capturado hasta ahora
        avg_sigma = np.mean(noise_buffer[i]) if noise_buffer[i] else 0.0
        ax.axhline(y=np.mean(raw_data[i, 100:]), color='gray', linestyle='--')
        ax.text(len(raw_data[i, 100:]) - 500, np.mean(raw_data[i, 100:]), 
                f'σ{i+1}: {avg_sigma:.4f}', color=plt.cm.tab10(i))

    total_captured = len(noise_buffer[0])
    ax.set_title(f"MODO CALIBRACIÓN | Muestras capturadas: {total_captured}/{SAMPLES_TO_AVERAGE}")
    ax.legend()
    ax.grid(True)
    
    # Si hemos capturado suficientes muestras, guarda y detén
    if total_captured >= SAMPLES_TO_AVERAGE:
        final_sigmas = [np.mean(noise_buffer[i]) for i in range(N_SENSORS)]
        
        np.save(NOISE_STATS_FILE, np.array(final_sigmas))
        print(f"\n✅ CALIBRACIÓN FINALIZADA.")
        print(f"   Valores Finales (Sigma): {final_sigmas}")
        
        # Cierra la figura y sale
        plt.close(fig) # <<--- CORRECCIÓN
        sys.exit() 
    

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == '__main__':
    noise_buffer = [[] for _ in range(N_SENSORS)]
    ser = None
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ani = animation.FuncAnimation(
            fig,
            update_plot,
            fargs=(ser, ax, noise_buffer, fig), # <<-- PASAMOS 'fig'
            interval=500, 
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        # Se asegura que la excepción se muestre antes de salir
        print(f"Error inesperado: {e}")
        if ser and ser.is_open: ser.close()
        sys.exit()