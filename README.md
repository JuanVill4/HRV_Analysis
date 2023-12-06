# Descripción

Este repositorio contiene los códigos para poder llevar a cabo un análisis de imagen extraída de un vídeo que tiene el objetivo último de obtener información relevante de señal cardiaca.

## Librerías necesarias

Para poder correr este código es necesario que se tenga python corriendo en su máquina y aunado a eso, es imperativo tener instaladas las siguientes librerías:

- **cv2**: OpenCV (Open Source Computer Vision Library) es una biblioteca de funciones de programación principalmente destinada a la visión por computadora en tiempo real.
  
- **numpy**: Una biblioteca para el lenguaje de programación Python, que agrega soporte para grandes matrices y arreglos multidimensionales, junto con una gran colección de funciones matemáticas de alto nivel para operar en estos arreglos.
  
- **pandas**: Una biblioteca de software para la manipulación y análisis de datos. Proporciona estructuras de datos y funciones necesarias para manipular datos estructurados.
  
- **matplotlib**: Una biblioteca de trazado para el lenguaje de programación Python y su extensión de matemáticas numéricas NumPy. Proporciona una API orientada a objetos para incrustar trazados en aplicaciones.
  
- **scipy**: Una biblioteca utilizada para la computación científica y técnica. Contiene módulos para optimización, álgebra lineal, integración, interpolación, funciones especiales, FFT, procesamiento de señales e imágenes, solucionadores de EDO y otras tareas comunes en ciencia e ingeniería.

Para Windows:
Si no se tiene Python instalado en tu máquina, ve a la página <https://www.python.org/downloads/> y haz todo el proceso de instalación.
Verifica además si ya se tiene **pip**, el gestor de paquetes para Python, usualmente al descargar python se incluye pero verifica con el comando: pip --version
En caso de no tenerlo instalado entonces dirígete a la siguiente página. <https://bootstrap.pypa.io/get-pip.py>  
Guarda la página como "get-pip.py". En la consola de comandos dírigete a la carpeta donde se encuentra ubicado. Típicamente sería con "cd Downloads" en Windows, y ejecuta "python get-pip.py". Con esto tendrás instalada la ultima versión de pip.

Una vez que se tengan todos estos requisitos, ve al directorio donde se guardó este repositorio y da doble clic sobre el archivo llamado <libraries.bat>, entonces se instalarán automáticamente.

### Carga del video

En el segundo bloque del Notebook **HRV_analysis and Visualization** se realiza la carga del video que se va a analizar. Este video tiene que ser descargado desde la base de datos común y ser colocado en la carpeta del repositorio, se añadirá un .gitignore para que cualquier cambio posterior ignore a los videos ya que estos no pueden ser cargados. En este segundo bloque, se debe de cambiar el nombre del archivo acorde al video que se quiera analizar que esté dentro del repositorio local.
