import cv2
import numpy as np

# Rutas a los archivos de los modelos de Deep Learning
PROTO_EDAD = "age_deploy.prototxt"
MODELO_EDAD = "age_net.caffemodel"
PROTO_GENERO = "gender_deploy.prototxt"
MODELO_GENERO = "gender_net.caffemodel"
MODELO_DETECTOR_CARAS = "haarcascade_frontalface_default.xml"

# Listas con los posibles resultados de edad y género
LISTA_EDAD = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
LISTA_GENERO = ['Fachero', 'Fachera']

# Valores medios de píxeles para el preprocesamiento de la imagen
VALORES_MEDIOS_MODELO = (78.4263377603, 87.7689143744, 114.895847746)

# Configuración del logo
RUTA_LOGO = "cpf-logo.png"
ANCHO_LOGO = 150  # Ancho deseado del logo

# Colores que usamos en la pantalla
COLOR_TEXTO = (255, 255, 255) # Blanco
COLOR_RECTANGULO_CARA = (255, 0, 0) # Azul

# --- 2. CARGA DE MODELOS Y ARCHIVOS ---
try:
    # Cargar el clasificador de rostros
    cascada_caras = cv2.CascadeClassifier(cv2.data.haarcascades + MODELO_DETECTOR_CARAS)

    # Cargar los modelos de la red neuronal para edad y género
    red_edad = cv2.dnn.readNet(MODELO_EDAD, PROTO_EDAD)
    red_genero = cv2.dnn.readNet(MODELO_GENERO, PROTO_GENERO)

    # Cargar la imagen del logo
    logo = cv2.imread(RUTA_LOGO, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"Error: No se pudo cargar el logo desde '{RUTA_LOGO}'.")
    
    # Redimensionar el logo manteniendo su relación de aspecto
    altura_logo_original, ancho_logo_original = logo.shape[:2]
    relacion_aspecto = ancho_logo_original / altura_logo_original
    nueva_altura_logo = int(ANCHO_LOGO / relacion_aspecto)
    logo = cv2.resize(logo, (ANCHO_LOGO, nueva_altura_logo))
    
    altura_logo, ancho_logo, canales_logo = logo.shape

except Exception as e:
    print(f"Ocurrió un error al cargar los recursos: {e}")
    print("Asegúrate de que todos los archivos estén en la misma carpeta.")
    exit()

#INICIALIZACIÓN DE LA CÁMARA
captura_video = cv2.VideoCapture(0)

if not captura_video.isOpened():
    print("No se pudo iniciar la cámara.")
    exit()

while True:
    # Capturar fotograma por fotograma de la cámara
    ret, fotograma = captura_video.read()
    if not ret:
        print("No se pudo leer el fotograma de la cámara. Saliendo del programa.")
        break

    # Voltear horizontalmente la imagen para efecto espejo
    fotograma = cv2.flip(fotograma, 1)

    # Convertir el fotograma a escala de grises para la detección de rostros 
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

    # Detectar los rostros en la imagen
    caras = cascada_caras.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Iterar sobre cada rostro detectado
    for (x, y, w, h) in caras:
        # Dibujar un rectángulo azul alrededor del rostro
        cv2.rectangle(fotograma, (x, y), (x+w, y+h), COLOR_RECTANGULO_CARA, 2)
        
        # Extraer solo la región del rostro para el análisis
        imagen_cara = fotograma[y:y+h, x:x+w].copy()

        # Preparar la imagen para los modelos de edad y género
        # Un 'blob' es una imagen preprocesada para una red neuronal
        blob = cv2.dnn.blobFromImage(imagen_cara, 1.0, (227, 227), VALORES_MEDIOS_MODELO, swapRB=False)
        
        # Predecir el género
        red_genero.setInput(blob)
        predicciones_genero = red_genero.forward()
        genero = LISTA_GENERO[predicciones_genero[0].argmax()]
        
        # Predecir la edad
        red_edad.setInput(blob)
        predicciones_edad = red_edad.forward()
        edad = LISTA_EDAD[predicciones_edad[0].argmax()]
        
        # Crear el texto para mostrar en la pantalla
        texto_superpuesto = f"{genero}, {edad}"
        
        # Colocar el texto blanco encima del rostro
        cv2.putText(fotograma, texto_superpuesto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXTO, 2, cv2.LINE_AA)

    # Posición para el logo en la esquina superior izquierda
    x_offset, y_offset = 10, 10
    
    # Asegurar que el logo quepa en el fotograma antes de pegarlo
    if y_offset + altura_logo <= fotograma.shape[0] and x_offset + ancho_logo <= fotograma.shape[1]:
        # Si el logo tiene transparencia 
        if canales_logo == 4:
            alpha_s = logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                fotograma[y_offset:y_offset+altura_logo, x_offset:x_offset+ancho_logo, c] = \
                    (alpha_s * logo[:, :, c] + alpha_l * fotograma[y_offset:y_offset+altura_logo, x_offset:x_offset+ancho_logo, c])
        else:
            # Si no tiene transparencia, simplemente copia la imagen
            fotograma[y_offset:y_offset+altura_logo, x_offset:x_offset+ancho_logo] = logo[:, :, :3]

    cv2.imshow('Deteccion de Edad y Genero', fotograma)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#LIBERAR RECURSOS Y CERRAR VENTANAS
captura_video.release()
cv2.destroyAllWindows()
