# ------Import standard libraries------
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft
from numpy.fft import fft

# import the cv2_imshow function  from patches to use in Google Collab
# import functions from the scipy.signal module for signal processing
from scipy.signal import butter, filtfilt, find_peaks, hamming
import scipy.signal as signal
#import and mount the drive module to access files and directories from Google Drive
import pandas as pd
import os
import glob

def generar_rutas_videos(directorio_videos, flag_dedo=True):
    """
    Genera las rutas de los archivos de video en el directorio especificado,
    filtrando por la presencia o ausencia del prefijo 'dedo'.
    
    Args:
    - directorio_base (str): El directorio donde se encuentran los archivos de video.
    - flag_dedo (bool): True para buscar videos con el prefijo 'dedo', False para buscar videos sin el prefijo.
    
    Returns:
    - list: Una lista con las rutas de los archivos de video que cumplen con el criterio.
    """
    # Usar glob para encontrar todos los archivos .mp4
    patron_busqueda = os.path.join(directorio_videos, "*.mp4")
    rutas_totales = glob.glob(patron_busqueda)

    # Filtrar las rutas según el flag_dedo
    if flag_dedo:
        # Incluir solo archivos que contienen '_dedo' en el nombre
        rutas = [ruta for ruta in rutas_totales if "_dedo.mp4" in os.path.basename(ruta)]
    else:
        # Excluir archivos que contienen '_dedo' en el nombre
        rutas = [ruta for ruta in rutas_totales if "_dedo.mp4" not in os.path.basename(ruta)]
    
    # Imprimir las rutas encontradas para verificar
    for ruta in rutas:
        print(ruta)
    
    print(f"Total de videos encontrados: {len(rutas)}")
    return rutas

def obtener_nombre_archivo(ruta):
    """
    Extrae el nombre del archivo sin la extensión de una ruta completa.
    
    Args:
    - ruta (str): La ruta completa del archivo.
    
    Returns:
    - str: El nombre del archivo sin la extensión.
    """
    # Extraer el nombre del archivo con la extensión
    nombre_con_extension = os.path.basename(ruta)
    
    # Separar la extensión y devolver solo el nombre
    nombre_sin_extension = os.path.splitext(nombre_con_extension)[0]
    
    return nombre_sin_extension

def arrays_name(directorio, nombre_base):
        # Crear los nombres de archivo completos
    saturation_filename = os.path.join(directorio, f"{nombre_base}_saturation_array.npy")
    hue_filename = os.path.join(directorio, f"{nombre_base}_hue_array.npy")
    return  hue_filename, saturation_filename

def guardar_arrays(directorio, nombre_base, saturation_array, hue_array):
    """
    Guarda los arrays de saturación y tono en el directorio especificado con un nombre de archivo basado en nombre_base.
    
    Args:
    - directorio (str): Directorio donde se guardarán los archivos.
    - nombre_base (str): Nombre base del archivo para usar en el nombre del archivo guardado.
    - saturation_array (numpy.ndarray): Array de saturación para guardar.
    - hue_array (numpy.ndarray): Array de tono para guardar.
    """
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    hue_filename, saturation_filename = arrays_name(directorio, nombre_base)
    
    # Guardar los arrays
    np.save(saturation_filename, saturation_array)
    np.save(hue_filename, hue_array)
    
    print(f"Archivos guardados: {saturation_filename} y {hue_filename}")

def hsv_arrays_name(directorio, nombre_base):
        # Crear los nombres de archivo completos
    hsv_filename = os.path.join(directorio, f"{nombre_base}_hsv_data.npz")
    return  hsv_filename

def rgb_arrays_name(directorio, nombre_base):
        # Crear los nombres de archivo completos
    rgb_filename = os.path.join(directorio, f"{nombre_base}_rgb_data.npz")
    return  rgb_filename

def save_hsv_arrays(directorio, nombre_base, hue_array, saturation_array, value_array):
    """
    
    """
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    hsv_filename = hsv_arrays_name(directorio, nombre_base)

    # Guardar los arrays
    np.savez(hsv_filename, hue=hue_array, saturation=saturation_array, value=value_array) 

    print(f"Archivos guardados: {hsv_filename}")

def save_rgb_arrays(directorio, nombre_base, red_array, green_array, blue_array):
    """
    
    """
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    rgb_filename = rgb_arrays_name(directorio, nombre_base)
    # Guardar los arrays
    np.savez(rgb_filename, red=red_array, green=green_array, blue=blue_array)

    print(f"Archivos guardados: {rgb_filename}")
    


def finger_hsv(video_filename):
    cap = cv2.VideoCapture(video_filename)

    #Para video de dedo
    saturation_values = []  # Lista para almacenar los valores de saturación
    hue_values = []  # Lista para almacenar los valores de tono
    value_values = []  # Lista para almacenar los valores de brillo

    red_values = []  # Lista para almacenar los valores de rojo
    green_values = []  # Lista para almacenar los valores de verde
    blue_values = []  # Lista para almacenar los valores de azul
    
    # Leer el video fotograma a fotograma
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Salir del bucle si no quedan más fotogramas

        # Convertir el fotograma completo a HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extraer los canales
        h, s , v = cv2.split(hsv_frame)

        b, g, r = cv2.split(frame)

        # Calcular la intensidad promedio del canal de saturación en el fotograma completo
        saturation_values.append(np.mean(s))
        # Calcular la intensidad promedio del canal de tono en el fotograma completo
        hue_values.append(np.mean(h))
        # Calcular la intensidad promedio del canal de brillo en el fotograma completo
        value_values.append(np.mean(v))

        # Calcular la intensidad promedio del canal de rojo en el fotograma completo
        red_values.append(np.mean(r))
        # Calcular la intensidad promedio del canal de verde en el fotograma completo
        green_values.append(np.mean(g))
        # Calcular la intensidad promedio del canal de azul en el fotograma completo
        blue_values.append(np.mean(b))



    # Convertir la lista de valores de saturación a un array de numpy
    saturation_array = np.array(saturation_values)
    # Convertir la lista de valores de tono a un array de numpy
    hue_array = np.array(hue_values)
    # Convertir la lista de valores de brillo a un array de numpy
    value_array = np.array(value_values)
    # Convertir la lista de valores de rojo a un array de numpy
    red_array = np.array(red_values)
    # Convertir la lista de valores de verde a un array de numpy
    green_array = np.array(green_values)
    # Convertir la lista de valores de azul a un array de numpy
    blue_array = np.array(blue_values)

    cap.release()
    cv2.destroyAllWindows()

    return hue_array, saturation_array, value_array, red_array, green_array, blue_array

def face_hsv(video_filename, face_cascade):
    
    cap = cv2.VideoCapture(video_filename)
    #Para video de rostro
    saturation_values = []  # Lista para almacenar los valores de saturación
    hue_values = []  # Lista para almacenar los valores de tono
    # Leer el video fotograma a fotograma
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Salir del bucle si no quedan más fotogramas

        # Convertir a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.1, 7)

        if len(faces) > 0:
            # Tomar el primer rostro detectado
            x, y, w, h = faces[0]

            # Definir una ROI para el rostro entero
            roi = frame[y:y+h, x:x+w]

            # Convertir la ROI a HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_roi)

            h = cv2.normalize(h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            s = cv2.normalize(s, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Calcular la intensidad promedio del canal de saturación en la ROI
            saturation_values.append(np.mean(s))

            # Calcular la intensidad promedio del canal de tono en la ROI
            hue_values.append(np.mean(h))

    # Convertir la lista de valores de saturación a un array de numpy
    saturation_array = np.array(saturation_values)
    # Convertir la lista de valores de tono a un array de numpy
    hue_array = np.array(hue_values)

    cap.release()
    cv2.destroyAllWindows()

    return hue_array, saturation_array

def rutas_csv(directorio_csv):
    """
    Genera las rutas de los archivos de video en el directorio especificado,
    filtrando por la presencia o ausencia del prefijo 'dedo'.
    
    Args:
    - directorio_base (str): El directorio donde se encuentran los archivos de video.
    - flag_dedo (bool): True para buscar videos con el prefijo 'dedo', False para buscar videos sin el prefijo.
    
    Returns:
    - list: Una lista con las rutas de los archivos de video que cumplen con el criterio.
    """
    # Usar glob para encontrar todos los archivos .mp4
    patron_busqueda = os.path.join(directorio_csv, "*.csv")
    rutas = glob.glob(patron_busqueda)
    
    # Imprimir las rutas encontradas para verificar
    for ruta in rutas:
        print(ruta)
    
    print(f"Total de videos encontrados: {len(rutas)}")
    return rutas

def rutas_arrays(directorio_arrays, nombre_base):
    """
    Busca y devuelve las rutas de los archivos Numpy que coinciden con un nombre base específico y contienen
    'hue_array.npy' o 'saturation_array.npy' en el directorio especificado.
    
    Args:
    - directorio_arrays (str): El directorio donde se encuentran los archivos.
    - nombre_base (str): Nombre base del archivo, como '001', para filtrar los archivos relevantes.
    
    Returns:
    - tuple: Tuplas que contienen listas de rutas para archivos de hue y saturation.
    """
    # Construir patrones de búsqueda para los archivos de hue y saturation
    patron_busqueda_hue = os.path.join(directorio_arrays, f"{nombre_base}*hue_array.npy")
    patron_busqueda_saturation = os.path.join(directorio_arrays, f"{nombre_base}*saturation_array.npy")
    
    # Usar glob para encontrar todos los archivos que coinciden con el patrón
    ruta_hue = glob.glob(patron_busqueda_hue)
    ruta_saturation = glob.glob(patron_busqueda_saturation)

    # Imprimir las rutas encontradas para verificar
    print("Archivos de Hue encontrados:")
    print(ruta_hue)

    print("Archivos de Saturation encontrados:")
    print(ruta_saturation)
        
    print(f"Total de archivos encontrados: {len(ruta_hue) + len(ruta_saturation)}")
    
    #si la lista esta vacia entonces regresa toda la lista, si tiene elementos regresa solo el primer elemento
    if len(ruta_hue) == 0:
        return ruta_hue, ruta_saturation
    else:
        return ruta_hue[0], ruta_saturation[0]
    
def hsv_file_path(directorio_arrays, nombre_base):
    """

    """
    # Construir patrones de búsqueda para los archivos de hue y saturation
    patron_busqueda_hsv = os.path.join(directorio_arrays, f"{nombre_base}*hsv_data.npz")
    
    # Usar glob para encontrar todos los archivos que coinciden con el patrón
    ruta_hsv = glob.glob(patron_busqueda_hsv)

    # Imprimir las rutas encontradas para verificar
    print("Archivos de HSV encontrados:")
    print(ruta_hsv)
        
    print(f"Total de archivos encontrados: {len(ruta_hsv)}")
    
    #si la lista esta vacia entonces regresa toda la lista, si tiene elementos regresa solo el primer elemento
    if len(ruta_hsv) == 0:
        return ruta_hsv
    else:
        return ruta_hsv[0]
    
def rgb_file_path(directorio_arrays, nombre_base):
    """

    """
    # Construir patrones de búsqueda para los archivos de hue y saturation
    patron_busqueda_rgb = os.path.join(directorio_arrays, f"{nombre_base}*rgb_data.npz")
    
    # Usar glob para encontrar todos los archivos que coinciden con el patrón
    ruta_rgb = glob.glob(patron_busqueda_rgb)

    # Imprimir las rutas encontradas para verificar
    print("Archivos de RGB encontrados:")
    print(ruta_rgb)
        
    print(f"Total de archivos encontrados: {len(ruta_rgb)}")
    
    #si la lista esta vacia entonces regresa toda la lista, si tiene elementos regresa solo el primer elemento
    if len(ruta_rgb) == 0:
        return ruta_rgb
    else:
        return ruta_rgb[0]
        


def average_arrays(array1, array2):
    """
    Calcula el promedio de dos arrays y devuelve un nuevo array con los valores promediados.
    
    Args:
    - array1 (numpy.ndarray): Primer array para promediar.
    - array2 (numpy.ndarray): Segundo array para promediar.
    
    Returns:
    - numpy.ndarray: Array con los valores promediados.
    """
    # Calcular el promedio de los dos arrays
    averaged_array = (array1 + array2) / 2
    return averaged_array

# Función para crear un filtro pasa banda Butterworth
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Función para aplicar el filtro
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def hr_analysis(hue_array, saturation_array, fs, lowcut, highcut):
    averaged_array = average_arrays(hue_array, saturation_array)
    # Filtrar la señal
    filtered_saturation = butter_bandpass_filter(saturation_array, lowcut, highcut, fs, order=5)
    filtered_hue = butter_bandpass_filter(hue_array, lowcut, highcut, fs, order=5)
    filtered_averaged = butter_bandpass_filter(averaged_array, lowcut, highcut, fs, order=5)

    # Aplicar una ventana de Hamming
    window_saturation = hamming(len(filtered_saturation))
    filtered_saturation = filtered_saturation * window_saturation

    window_hue = hamming(len(filtered_hue))
    filtered_hue = filtered_hue * window_hue

    window_averaged = hamming(len(filtered_averaged))
    filtered_averaged = filtered_averaged * window_averaged


    # Realizar FFT en la señal filtrada
    saturation_fft = fft(filtered_saturation)
    hue_fft = fft(filtered_hue)
    averaged_fft = fft(filtered_averaged)

    # Calcular las frecuencias correspondientes
    freqs_saturation = np.fft.fftfreq(len(filtered_saturation), 1/fs)
    freqs_hue = np.fft.fftfreq(len(filtered_hue), 1/fs)
    freqs_averaged = np.fft.fftfreq(len(filtered_averaged), 1/fs)

    # Encuentra la frecuencia con la magnitud más alta en el rango de la frecuencia cardíaca
    idx_saturation = np.argmax(np.abs(saturation_fft))
    pulse_freq_saturation = freqs_saturation[idx_saturation]
    heart_rate_saturation = abs(pulse_freq_saturation * 60)  # Convertir a latidos por minuto

    idx_hue = np.argmax(np.abs(hue_fft))
    pulse_freq_hue = freqs_hue[idx_hue]
    heart_rate_hue = abs(pulse_freq_hue * 60)  # Convertir a latidos por minuto

    idx_averaged = np.argmax(np.abs(averaged_fft))
    pulse_freq_averaged = freqs_averaged[idx_averaged]
    heart_rate_averaged = abs(pulse_freq_averaged * 60)  # Convertir a latidos por minuto

    return heart_rate_hue, heart_rate_saturation, heart_rate_averaged

def hr_analysis_rgb(red_array, green_array, blue_array, fs, lowcut, highcut):
    # Filtrar la señal
    filtered_red = butter_bandpass_filter(red_array, lowcut, highcut, fs, order=5)
    filtered_green = butter_bandpass_filter(green_array, lowcut, highcut, fs, order=5)
    filtered_blue = butter_bandpass_filter(blue_array, lowcut, highcut, fs, order=5)

    # Aplicar una ventana de Hamming
    # window_red = hamming(len(filtered_red))
    # filtered_red = filtered_red * window_red

    # window_green = hamming(len(filtered_green))
    # filtered_green = filtered_green * window_green

    # window_blue = hamming(len(filtered_blue))
    # filtered_blue = filtered_blue * window_blue


    # Realizar FFT en la señal filtrada
    red_fft = fft(filtered_red)
    green_fft = fft(filtered_green)
    blue_fft = fft(filtered_blue)

    # Calcular las frecuencias correspondientes
    freqs_red = np.fft.fftfreq(len(filtered_red), 1/fs)
    freqs_green = np.fft.fftfreq(len(filtered_green), 1/fs)
    freqs_blue = np.fft.fftfreq(len(filtered_blue), 1/fs)

    # Encuentra la frecuencia con la magnitud más alta en el rango de la frecuencia cardíaca
    idx_red = np.argmax(np.abs(red_fft))
    pulse_freq_red = freqs_red[idx_red]
    heart_rate_red = abs(pulse_freq_red * 60)  # Convertir a latidos por minuto

    idx_green = np.argmax(np.abs(green_fft))
    pulse_freq_green = freqs_green[idx_green]
    heart_rate_green = abs(pulse_freq_green * 60)  # Convertir a latidos por minuto

    idx_blue = np.argmax(np.abs(blue_fft))
    pulse_freq_blue = freqs_blue[idx_blue]
    heart_rate_blue = abs(pulse_freq_blue * 60)  # Convertir a latidos por minuto

    return heart_rate_red, heart_rate_green, heart_rate_blue


def hr_error_hsv(csv_file, heart_rate_hue, heart_rate_saturation, heart_rate_averaged):
    data_video= pd.read_csv(csv_file)

    # extract the data from the PULSE column
    pulse_signal = data_video['PULSE']

    # Take only the numerical values of the signal
    pulse_signal = pd.to_numeric(pulse_signal, errors='coerce')

    # print the size of the signal
    print("Signal size:", pulse_signal.size)

    # calculate and print the average value of the Heart Rate
    heart_rate_csv = np.mean(pulse_signal)
    print("\nHeart rate in csv: {:.2f} bpm".format(heart_rate_csv))

    # calculate and print the error
    error_saturation=abs(heart_rate_saturation-heart_rate_csv)
    print("\nError saturación: {:.2f} bpm".format(error_saturation))

    error_hue=abs(heart_rate_hue-heart_rate_csv)
    print("\nError tono: {:.2f} bpm".format(error_hue))

    error_averaged=abs(heart_rate_averaged-heart_rate_csv)
    print("\nError promedio: {:.2f} bpm".format(error_averaged))

    return heart_rate_csv, error_saturation, error_hue, error_averaged

def hr_error_rgb(csv_file, heart_rate_red, heart_rate_green, heart_rate_blue):
    data_video= pd.read_csv(csv_file)

    # extract the data from the PULSE column
    pulse_signal = data_video['PULSE']

    # Take only the numerical values of the signal
    pulse_signal = pd.to_numeric(pulse_signal, errors='coerce')

    # print the size of the signal
    print("Signal size:", pulse_signal.size)

    # calculate and print the average value of the Heart Rate
    heart_rate_csv = np.mean(pulse_signal)
    print("\nHeart rate in csv: {:.2f} bpm".format(heart_rate_csv))

    # calculate and print the error
    error_red=abs(heart_rate_red-heart_rate_csv)
    print("\nError saturación: {:.2f} bpm".format(error_red))

    error_green=abs(heart_rate_green-heart_rate_csv)
    print("\nError tono: {:.2f} bpm".format(error_green))

    error_blue=abs(heart_rate_blue-heart_rate_csv)
    print("\nError promedio: {:.2f} bpm".format(error_blue))

    return heart_rate_csv, error_red, error_green, error_blue