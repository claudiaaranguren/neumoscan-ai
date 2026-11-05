# NeumoScan AI · Tamizaje de neumonía en radiografías de tórax pediátricas

**Proyecto ACA 2 · Aprendizaje Automático** · Especialización en Inteligencia Artificial (CUN)

- **Autora:** Claudia Teresa Aranguren Peña
- **Curso:** APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4
- **Docente:** Dr. Hamilton Rivera Flor

## Enlaces

- Notebook (Google Colab): https://colab.research.google.com/drive/1hxoqbGEsOEeYZuXEvzU_OFGRBXrHKljj?usp=sharing
- Aplicación (Streamlit Community Cloud): https://neumoscan-ai-ctap.streamlit.app
- Repositorio (GitHub): https://github.com/claudiaaranguren/neumoscan-ai
- Video de sustentación: se completa al publicar

## Qué hace

Clasifica radiografías de tórax pediátricas como *normal* o *neumonía* con una red neuronal
convolucional (MobileNetV2 con *transfer learning*, TensorFlow/Keras), entrenada con el dataset
Chest X-Ray Images (Pneumonia). Los datos se gestionaron en SQLite y se dividieron por paciente
para evitar fuga de información. Con un umbral clínico de 0.35, en el conjunto de prueba
(919 radiografías) obtuvo ROC-AUC = 0.987, Recall = 0.961 y especificidad = 0.944.
La app muestra la probabilidad y un mapa de calor Grad-CAM de dónde miró la red.
Herramienta académica de tamizaje: no reemplaza el diagnóstico médico.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `app.py` | Aplicación de Streamlit |
| `requirements.txt` | Librerías que instala el servidor |
| `modelo_neumonia.keras` | Red neuronal entrenada (incluye el preprocesamiento) |
| `metricas_cnn.json` | Métricas, umbral, curvas de aprendizaje, ROC y particiones |
| `gradcam_ejemplos.png` | Figura de interpretabilidad (Grad-CAM) |
| `radiografias.db` | Base de datos SQLite: inventario, pacientes y partición de cada radiografía |
| `ejemplos/` | Radiografías de prueba para usar la app |

## Ejecutar en local

```
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud se usa Python 3.12 (Advanced settings), compatible con TensorFlow.

## Referencias

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., ... Zhang, K. (2018).
Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5),
1122–1131. https://doi.org/10.1016/j.cell.2018.02.010

Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)* [Conjunto de datos]. Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals
and linear bottlenecks. En *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474
