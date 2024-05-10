import os
from functions import generar_rutas_videos, obtener_nombre_archivo, arrays_name, face_hsv, guardar_arrays
import cv2
directorio_videos = "DB"
directorio_arrays = "Face_arrays"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

rutas_cara = generar_rutas_videos(directorio_videos, flag_dedo=False)  # Debería devolver rutas con 'dedo'

for ruta_cara in rutas_cara:
    nombre_base = obtener_nombre_archivo(ruta_cara)
    hue_filename, saturation_filename = arrays_name(directorio_arrays, nombre_base)
    #verifica si existe el archivo y si no indica que no existe
    if not os.path.exists(hue_filename) and not os.path.exists(saturation_filename):
        print(f"El archivo {hue_filename} NO existe y el archivo {saturation_filename} NO existe")
        hue_array, saturation_array = face_hsv(ruta_cara, face_cascade)
        guardar_arrays(directorio_arrays, nombre_base, saturation_array, hue_array)
        
    else:
        print(f"Los siguientes archivos fueron encontrados: {hue_filename} y {saturation_filename}")