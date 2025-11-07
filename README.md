# Detección de rasgos biométricos

## Tareas:
### Tarea 1
#### Enunciado:
Proponer dos escenarios de aplicación y desarrollar dos prototipos de temática libre que provoquen reacciones a partir de la información extraída del rostro. Uno de los prototipos deberá incluir el uso de algún modelo entrenado por ustedes para la extracción de información biometríca, similar al ejemplo del género planteado durante la práctica pero con diferente aplicación (emociones, raza, edad...).

#### Resolución:
Decidimos utilizar este para realizar la tarea libre, 

---

## Tecnologías utilizadas

- **Python 3.11+**  
- **OpenCV** → Lectura y escritura de vídeos, visualización de resultados.  
- **Ultralytics YOLO (v8 / v11)** → Detección y tracking de objetos.  
- **Torch / Torch DirectML** → Aceleración con GPU AMD o CPU.  
- **CSV / defaultdict** → Registro de resultados.  


##  1. Dataset

Se descarga un **conjunto de datos de matrículas** en formato YOLO desde [Kaggle](https://www.kaggle.com/), mediante la librería `kagglehub`.

```python
import kagglehub
path = kagglehub.dataset_download("sujaymann/car-number-plate-dataset-yolo-format")
print("Path to dataset files:", path)
````

Este dataset sirve para **entrenar un modelo YOLO personalizado** que será capaz de detectar matrículas en imágenes reales.



##  2. Detección y seguimiento de personas y vehículos

En este bloque se crea un **prototipo de detección y seguimiento** de personas y coches en vídeos propios, utilizando el modelo **YOLOv11n** y el método `track()`.

* Se guardan los resultados en un vídeo anotado (`salida_yolo_tracking_sinOCR.mp4`). El video resultante `VC_P4.mp4`.
* Se genera un archivo CSV (`detecciones_yolo_tracking_sinOCR.csv`) con los objetos detectados y su ID de seguimiento.


##  3. Entrenamiento de un modelo YOLO personalizado

Se entrena un modelo YOLO para detectar **únicamente matrículas** usando el dataset descargado.
El archivo `data.yaml` define las rutas de entrenamiento, validación y las clases disponibles.

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    imgsz=416,
    epochs=10,
    batch=4,
    device="mps",
    name="matriculas_detector2"
)
```
Los resultados se encuentran en la carpeta `/runs`.


##  4. Detección de vehículos en vídeos propios

Se utiliza el modelo YOLO para detectar **vehículos en movimiento** y extraer su región inferior,
donde generalmente se ubica la matrícula.

Si está disponible **EasyOCR**, se realiza lectura preliminar de las matrículas detectadas.
El proceso genera:

* `salida_simple.mp4` → vídeo anotado
* `detecciones_simple.csv` → resultados

---

## 5. Lectura de matrículas detectadas con EasyOCR

En esta etapa se aplican técnicas de OCR sobre las imágenes generadas por el detector YOLO.
Para cada imagen:

* Se recorta la región de la matrícula.
* Se mejora el contraste y se aplica **EasyOCR**.
* Los resultados se guardan en `lecturas_matriculas.csv`.


---

## 6. Lectura de matrículas detectadas con Tesseract OCR

Se repite el proceso anterior usando **Tesseract**, aplicando un preprocesado más avanzado (filtro bilateral, ecualización y binarización).

Se guarda cada lectura en `lecturas_matriculas_tesseract.csv` junto con el tiempo medio de inferencia.


## 7. Evaluación de precisión y similitud

Los resultados de ambos OCRs se comparan en base a:

* **Precisión exacta (%)** → lecturas que coinciden exactamente con la matrícula real.
* **Similitud media (%)** → semejanza entre la lectura y la matrícula real usando la distancia de Levenshtein.

Ejemplo de resultados en consola:

```
Comparativa OCR de matrículas
================================
Imágenes evaluadas: 30

EasyOCR
 - Precisión exacta: 0.00%
 - Similitud media:  20.72%

Tesseract
 - Precisión exacta: 0.00%
 - Similitud media:  16.67%

Modelo con más aciertos: Empate
```

---

##  8. Gráfica comparativa de rendimiento de OCRs

Finalmente, se genera una **gráfica comparativa** (`comparativa_ocr.png`) que representa:

* En barras: la **similitud media (%)** de EasyOCR y Tesseract.
* En línea naranja: el **tiempo medio de inferencia (ms)**.

Esta gráfica permite visualizar de forma conjunta el equilibrio entre **precisión** y **velocidad** de ambos OCRs.

---

## Resultado

![Comparativa de OCRs](comparativa_ocr.png)

## Instalacion:
```bash
conda create --name VC_P5 python=3.11.5
conda activate VC_P5
pip install opencv-python
pip install matplotlib
pip install imutils
pip install mtcnn
pip install tensorflow   
pip install deepface
pip install tf-keras
pip install cmake
pip install dlib
```