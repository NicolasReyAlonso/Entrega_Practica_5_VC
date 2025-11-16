# Práctica 5: Detección y Caracterización de Caras

## Descripción General
Esta práctica implementa dos prototipos que utilizan técnicas de visión por computadora para detectar y reaccionar a información extraída del rostro humano en tiempo real.

## Prototipos Desarrollados

### 1. **Prototipo 1: Clasificador de Emociones con Filtros Visuales**
**Temática**: Modelo entrenado por nosotros para extracción de información biométrica

#### Características:
- **Modelo CNN personalizado** entrenado con dataset de emociones
- **Clasificación de 7 emociones**:
  - Enfado, Asco, Miedo, Feliz, Neutral, Triste, Sorpresa
- **Filtros visuales en tiempo real** según emoción detectada
- **Validación cruzada** (k-fold) para robustez del modelo

#### Dataset Utilizado

### **Face Expression Recognition Dataset**

#### Fuente:
- **Plataforma**: Kaggle
- **Creador**: jonathanoheix
- **Enlace**: [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

#### Características del Dataset:
- **Categorías**: 7 emociones diferentes
- **Resolución**: 48×48 píxeles
- **Formato de color**: Escala de grises

#### Arquitectura del Modelo:
```python
CNN con:
- 2 capas convolucionales (32 y 64 filtros)
- MaxPooling para reducción dimensional
- Capa Fully Connected con Dropout
- Salida Softmax para clasificación multiclase
```

#### Filtros Implementados:
- **Feliz**: Aumento de brillo y saturación
- **Triste**: Tonos azules y enfriamiento de imagen
- **Enfado**: Tonos rojos e intensificación


### 2. **Prototipo 2: Sistema de Expresiones Faciales con Efectos Visuales**
**Temática**: Completamente libre - Arte interactivo facial

#### Características:
- **Detección en tiempo real** usando MediaPipe Face Mesh
- **Dos expresiones detectadas**:
  - **Parpadeo**: Genera lágrimas animadas
  - **Boca abierta**: Emite partículas de fuego
- **Efectos visuales** con física básica (gravedad, turbulencia)
- **Métricas biométricas**:
  - EAR (Eye Aspect Ratio) para parpadeos
  - MAR (Mouth Aspect Ratio) para boca abierta

#### Tecnologías:
- MediaPipe para landmarks faciales
- OpenCV para procesamiento de video
- Algoritmos de relación de aspecto (EAR/MAR)
- Sistemas de partículas para efectos

---

## Instalación y Configuración

### Dependencias Principales
```bash
# Entorno base
conda create --name VC_P5 python=3.11.5
conda activate VC_P5

# Paquetes esenciales
pip install opencv-python
pip install mediapipe
pip install scipy
pip install scikit-learn
pip install tensorflow
pip install matplotlib
```
### Para el Prototipo 1 (Clasificación de Emociones)
```bash
pip install tensorflow
pip install kagglehub
pip install scikit-image
```

### Para el Prototipo 2 (Expresiones Faciales)
```bash
pip install mediapipe
```

## Uso de IA
Para el desarrollo de estos ejercicios hemos hecho uso de la herramienta de inteligencia artificial copilot integrada en vs code para preguntar sobre los erroresque nos iban surgiendo

## 👥 Autores
Wafa Azdad Triki y Nicolás Rey Alonso.



