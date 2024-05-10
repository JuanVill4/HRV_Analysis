import os
import logging
from functions import generar_rutas_videos, obtener_nombre_archivo, arrays_name, finger_hsv, guardar_arrays, rgb_arrays_name, hsv_arrays_name, save_hsv_arrays, save_rgb_arrays
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

directorio_videos = "DB"
directorio_hsv = "Finger_hsv"
directorio_rgb = "Finger_rgb"

rutas_dedo = generar_rutas_videos(directorio_videos, flag_dedo=True)  # Debería devolver rutas con 'dedo'

# for ruta_dedo in rutas_dedo:
#     nombre_base = obtener_nombre_archivo(ruta_dedo)
#     hue_filename, saturation_filename = arrays_name(directorio_arrays, nombre_base)
#     #verifica si existe el archivo y si no indica que no existe
#     if not os.path.exists(hue_filename) and not os.path.exists(saturation_filename):
#         print(f"El archivo {hue_filename} NO existe y el archivo {saturation_filename} NO existe")
#         hue_array, saturation_array = finger_hsv(ruta_dedo)
#         guardar_arrays(directorio_arrays, nombre_base, saturation_array, hue_array)
        
#     else:
#         print(f"Los siguientes archivos fueron encontrados: {hue_filename} y {saturation_filename}")


for ruta_dedo in rutas_dedo:
    nombre_base = obtener_nombre_archivo(ruta_dedo)
    rgb_filename, hsv_filename = rgb_arrays_name(directorio_rgb, nombre_base), hsv_arrays_name(directorio_hsv, nombre_base)
    #verifica si existe el archivo y si no indica que no existe
    if not os.path.exists(rgb_filename) and not os.path.exists(hsv_filename):
        logging.info(f"El archivo {rgb_filename} NO existe y el archivo {hsv_filename} NO existe")
        hue_array, saturation_array, value_array, red_array, green_array, blue_array = finger_hsv(ruta_dedo)
        save_hsv_arrays(directorio_hsv, nombre_base, hue_array, saturation_array, value_array)
        save_rgb_arrays(directorio_rgb, nombre_base, red_array, green_array, blue_array)

        logging.info(f"Los siguientes arrays fueron guardados: {rgb_filename} y {hsv_filename}")
        
    else:
        logging.info(f"Los siguientes archivos fueron encontrados: {rgb_filename} y {hsv_filename}")