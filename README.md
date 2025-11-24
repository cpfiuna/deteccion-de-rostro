# Detección de Rostro con Predicción de Edad y Género

Un sistema de detección de rostros en tiempo real que utiliza **OpenCV** y redes neuronales profundas para identificar caras y predecir la edad y género de las personas detectadas a través de la cámara web.

## 🚀 Características

- **Detección de rostros en tiempo real** usando clasificadores Haar Cascade
- **Predicción de edad** con 8 rangos etarios diferentes
- **Predicción de género** (Hombre/Mujer)
- **Interfaz visual en vivo** con cámara web
- **Logo personalizable** en la esquina superior izquierda
- **Efecto espejo** para una mejor experiencia de usuario

## 📋 Requisitos

### Dependencias de Python
```bash
pip install opencv-python numpy
```

### Archivos necesarios
El proyecto requiere los siguientes archivos (incluidos en el repositorio):

- `haarcascade_frontalface_default.xml` - Clasificador para detección de rostros
- `age_deploy.prototxt` - Arquitectura de la red neuronal para edad
- `age_net.caffemodel` - Modelo preentrenado para predicción de edad
- `gender_deploy.prototxt` - Arquitectura de la red neuronal para género
- `gender_net.caffemodel` - Modelo preentrenado para predicción de género
- `cpf-logo.png` - Logo personalizable (opcional)

## 🛠️ Instalación

1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/cpfiuna/deteccion-de-rostro.git
   cd deteccion-de-rostro
   ```

2. **Instala las dependencias:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Ejecuta la aplicación:**
   ```bash
   python detector_edad.py
   ```

## 📖 Uso

1. Al ejecutar el script, se abrirá una ventana mostrando la transmisión en vivo de tu cámara web
2. El sistema detectará automáticamente los rostros presentes en la imagen
3. Para cada rostro detectado, se mostrará:
   - Un **rectángulo azul** alrededor del rostro
   - **Texto blanco** con la predicción de género y edad encima del rostro
4. Presiona la tecla **'q'** para salir de la aplicación

## 🔧 Configuración

### Rangos de Edad
El sistema predice la edad en los siguientes rangos:
- (0-2) años
- (4-6) años  
- (8-12) años
- (15-20) años
- (25-32) años
- (38-43) años
- (48-53) años
- (60-100) años


### Configuración de Colores
Puedes modificar los colores en las siguientes variables:
- `COLOR_TEXTO`: Color del texto de predicción (por defecto: blanco)
- `COLOR_RECTANGULO_CARA`: Color del rectángulo alrededor del rostro (por defecto: azul)

## 🧠 Tecnologías Utilizadas

- **OpenCV**: Procesamiento de imágenes y visión por computadora
- **NumPy**: Operaciones matemáticas y manipulación de arrays
- **Redes Neuronales Convolucionales (CNN)**: Para predicción de edad y género
- **Clasificadores Haar Cascade**: Para detección de rostros
- **Modelos Caffe**: Framework de deep learning para las predicciones

## 📁 Estructura del Proyecto

```
deteccion-de-rostro/
├── detector_edad.py              # Script principal
├── haarcascade_frontalface_default.xml  # Clasificador de rostros
├── age_deploy.prototxt           # Arquitectura red neuronal (edad)
├── age_net.caffemodel           # Modelo entrenado (edad)
├── gender_deploy.prototxt       # Arquitectura red neuronal (género)
├── gender_net.caffemodel        # Modelo entrenado (género)
├── cpf-logo.png                 # Logo personalizable
├── LICENSE                      # Licencia del proyecto
└── README.md                    # Este archivo
```

## 🚨 Solución de Problemas

### Error: "No se pudo iniciar la cámara"
- Verifica que tu cámara web esté conectada y funcionando
- Asegúrate de que ninguna otra aplicación esté usando la cámara
- Prueba cambiar el índice de la cámara de `0` a `1` en la línea: `captura_video = cv2.VideoCapture(0)`

### Error: "No se pudo cargar el logo"
- Verifica que el archivo `cpf-logo.png` esté en el directorio del proyecto
- Si no tienes logo, puedes comentar las líneas relacionadas con el logo en el código

### Rendimiento lento
- Reduce el tamaño mínimo de detección modificando `minSize=(100, 100)` a un valor mayor
- Ajusta el `scaleFactor` para mejorar la velocidad de detección

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo `LICENSE`.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz un Fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 👥 Créditos

- Modelos de edad y género basados en trabajos de investigación en deep learning
- Clasificadores Haar Cascade de OpenCV
- Desarrollado por el equipo del Club de Programación FIUNA

---

**Nota**: Este sistema proporciona estimaciones basadas en modelos de machine learning y no debe considerarse como una medición exacta de edad o género.

---
## Contacto

<div align="center">

  [![Instagram](https://img.shields.io/badge/Instagram-cpf?style=plastic&logo=instagram&logoColor=%23fafafa&labelColor=%23FF0069&color=%23FF0069)](https://instagram.com/cpfiuna)
  [![X](https://img.shields.io/badge/X-cpf?style=plastic&logo=x&logoColor=%23fafafa&labelColor=%23000000&color=%23000000)](https://x.com/cpfiuna)
  [![Discord](https://img.shields.io/badge/Discord-cpf?style=plastic&logo=discord&logoColor=%23fafafa&labelColor=%235865F2&color=%235865F2)](https://discord.gg/UtRpKw2ay4)
  [![YouTube](https://img.shields.io/badge/YouTube-cpf?style=plastic&logo=youtube&logoColor=%23fafafa&labelColor=%23FF0000&color=%23FF0000)](https://youtube.com/@cpfiuna)
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-cpf?style=plastic&logo=inspire&logoColor=%23FAFAFA&labelColor=%230A66C2&color=%230A66C2)](https://www.linkedin.com/company/cpfiuna)

  Visitá nuestra [página web](https://cpfiuna.io) :)

</div>
