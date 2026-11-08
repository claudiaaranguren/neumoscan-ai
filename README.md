# NeumoScan AI · Tamizaje de neumonía en radiografías de tórax pediátricas

**Proyecto ACA 2 · Aprendizaje Automático** · Especialización en Inteligencia Artificial (CUN)

- **Autora:** Claudia Teresa Aranguren Peña
- **Curso:** APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4
- **Docente:** Dr. Hamilton Rivera Flor

## Enlaces

- Notebook (Google Colab): https://colab.research.google.com/drive/1hxoqbGEsOEeYZuXEvzU_OFGRBXrHKljj?usp=sharing
- Aplicación (Streamlit Community Cloud): https://neumoscan-ai-ctap.streamlit.app
- Repositorio (GitHub): https://github.com/claudiaaranguren/neumoscan-ai
- Video de sustentación (YouTube): https://youtu.be/UN7y_VpBnbs
- Póster (PDF): Poster_ACA2_NeumoScan.pdf en este repositorio

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
| `.streamlit/config.toml` | Tema de colores de la app |
| `Poster_ACA2_NeumoScan.pdf` | Póster tipo infografía |

## Ejecutar en local

```
pip install -r requirements.txt
streamlit run app.py
```

En Streamlit Community Cloud se usa Python 3.12 (Advanced settings), compatible con TensorFlow.
Reproducibilidad: TensorFlow 2.21 · semilla 42 · partición por paciente guardada en `radiografias.db`.

## Referencias

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G.,
Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J.,
Ziyar, I., . . . Zhang, K. (2018).
Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5),
1122–1131. https://doi.org/10.1016/j.cell.2018.02.010

Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)* [Conjunto de datos]. Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals
and linear bottlenecks. En *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual
explanations from deep networks via gradient-based localization. En *Proceedings of the IEEE International
Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74
