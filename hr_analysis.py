from functions import rutas_csv, obtener_nombre_archivo, rutas_arrays, hr_analysis, hr_error_rgb, rgb_file_path, hr_analysis_rgb
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

rutas_archivos = rutas_csv("DB")
lowcut = 1.1
highcut = 1.6
fs = 30
# resultados = pd.DataFrame(columns=[
#     'Nombre de Archivo', 'Ritmo Cardiaco Hue', 'Ritmo Cardiaco Saturación',
#     'Ritmo Cardiaco Promedio', 'Error Hue', 'Error Saturación', 'Error Promedio'
# ])

# for ruta in rutas_archivos:
#     nombre_archivo = obtener_nombre_archivo(ruta)
#     print(nombre_archivo)
#     ruta_array_hue, ruta_array_saturation = rutas_arrays("Finger_arrays", nombre_archivo)
#     #verificar si las listas no estan vacias
#     if ruta_array_hue and ruta_array_saturation:
#         print("Archivos encontrados")
#         hue_array = np.load(ruta_array_hue)
#         saturation_array = np.load(ruta_array_saturation)
#         heart_rate_hue, heart_rate_saturation, heart_rate_average = hr_analysis(hue_array, saturation_array, fs, lowcut, highcut)
#         print(f"El ritmo cardiaco calculado con el array de tono es: {heart_rate_hue}")
#         print(f"El ritmo cardiaco calculado con el array de saturación es: {heart_rate_saturation}")
#         print(f"El ritmo cardiaco promedio es: {heart_rate_average}")
#         heart_rate_csv, error_hue, error_saturation, error_average = hr_error(ruta, heart_rate_hue, heart_rate_saturation, heart_rate_average)

#         new_row = pd.DataFrame([{
#             'Nombre de Archivo': nombre_archivo,
#             'Ritmo Cardiaco Hue': heart_rate_hue,
#             'Ritmo Cardiaco Saturación': heart_rate_saturation,
#             'Ritmo Cardiaco Promedio': heart_rate_average,
#             'Error Hue': error_hue,
#             'Error Saturación': error_saturation,
#             'Error Promedio': error_average,
#             'Ritmo Cardiaco CSV': heart_rate_csv
#         }])

#         resultados = pd.concat([resultados, new_row], ignore_index=True)


#     else:
#         print("No se encontraron archivos")

# resultados.to_excel("resultados_ritmo_cardiaco.xlsx", index=False)

resultados = pd.DataFrame(columns=[
    'Nombre de Archivo', 'Ritmo Cardiaco Red', 'Ritmo Cardiaco Green',
    'Ritmo Cardiaco Blue', 'Ritmo Cardiaco CSV', 'Error Red', 'Error Green', 'Error Blue'
])

for ruta in rutas_archivos:
    nombre_archivo = obtener_nombre_archivo(ruta)
    logging.info(nombre_archivo)
    ruta_rgb = rgb_file_path("Finger_rgb", nombre_archivo)
    #verificar si las listas no estan vacias
    if ruta_rgb:
        logging.info("Archivos encontrados")
        rgb_data = np.load(ruta_rgb)

        red_array = rgb_data['red']
        green_array = rgb_data['green']
        blue_array = rgb_data['blue']

        heart_rate_red, heart_rate_green, heart_rate_blue = hr_analysis_rgb(red_array, green_array, blue_array, fs, lowcut, highcut)
        print(f"El ritmo cardiaco calculado con el canal rojo es: {heart_rate_red}")
        print(f"El ritmo cardiaco calculado con el canal verde es: {heart_rate_green}")
        print(f"El ritmo cardiaco calculado con el canal azul es: {heart_rate_blue}")
        heart_rate_csv, error_red, error_green, error_blue = hr_error_rgb(ruta, heart_rate_red, heart_rate_green, heart_rate_blue)

        new_row = pd.DataFrame([{
            'Nombre de Archivo': nombre_archivo,
            'Ritmo Cardiaco Red': heart_rate_red,
            'Ritmo Cardiaco Green': heart_rate_green,
            'Ritmo Cardiaco Blue': heart_rate_blue,
            'Ritmo Cardiaco CSV': heart_rate_csv,
            'Error Red': error_red,
            'Error Green': error_green,
            'Error Blue': error_blue
        }])


        resultados = pd.concat([resultados, new_row], ignore_index=True)

        rgb_data.close()


    else:
        logging.info("No se encontraron archivos")

resultados.to_excel("resultados_ritmo_cardiaco_rgb.xlsx", index=False)