"""
NeumoScan AI - Tamizaje de neumonía en radiografías de tórax pediátricas
Proyecto ACA 2 - Aprendizaje Automático (CUN)

Arquitectura de archivos:
  - modelo_neumonia.keras : red completa (aumento de datos + escalado + MobileNetV2 + cabeza densa).
                            Recibe la imagen 224x224x3 en píxeles 0-255 y devuelve P(neumonía).
  - metricas_cnn.json     : métricas, umbral, curvas de aprendizaje, ROC y particiones generadas
                            por el notebook. La app NO contiene números escritos a mano.
  - gradcam_ejemplos.png  : figura de interpretabilidad del notebook.
  - ejemplos/*.jpeg       : radiografías de prueba para usar la app sin imágenes propias.
"""

import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
st.set_page_config(
    page_title="NeumoScan AI | Tamizaje de neumonía",
    page_icon="⚕️",
    layout="wide",
)

# ENLACES DEL PROYECTO: pegue cada enlace entre comillas cuando lo tenga.
# Si queda vacío (""), la app muestra "Disponible pronto".
ENLACES = {
    "📓 Notebook (Google Colab)": "https://colab.research.google.com/drive/1hxoqbGEsOEeYZuXEvzU_OFGRBXrHKljj?usp=sharing",
    "💻 Repositorio (GitHub)": "https://github.com/claudiaaranguren/neumoscan-ai",
    "🎥 Video de sustentación": "",
    "📄 Póster (infografía PDF)": "",
}

ARCHIVO_MODELO = "modelo_neumonia.keras"
ARCHIVO_METRICAS = "metricas_cnn.json"
ARCHIVO_GRADCAM = "gradcam_ejemplos.png"
CARPETA_EJEMPLOS = "ejemplos"

# Paleta sobria de salud (la misma del proyecto ACA 1)
MARINO, AZUL, TEAL, VERDE = "#0B3C5D", "#1B6CA8", "#0B7A6B", "#26784A"
CORAL, AMBAR, PURPURA = "#C53030", "#9C5D06", "#5B4B8A"
COLOR_NORMAL, COLOR_NEUMONIA = AZUL, CORAL

st.markdown(f"""
<style>
html {{ font-size: 18px; }}
.tarjeta {{ border-radius: 16px; padding: 16px 12px; color: #FFFFFF; text-align: center;
           box-shadow: 0 6px 16px rgba(11, 60, 93, 0.22); margin-bottom: 12px; }}
.tarjeta .etq {{ font-size: 0.9rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
.tarjeta .val {{ font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin: 6px 0 4px; }}
.tarjeta .det {{ font-size: 0.9rem; }}
.titulo {{ background: linear-gradient(90deg, {MARINO}, {AZUL}); color: #FFFFFF; padding: 12px 20px;
          border-radius: 12px; border-left: 10px solid {TEAL}; font-size: 1.45rem; font-weight: 800;
          margin: 26px 0 16px; }}
.banner {{ border-radius: 16px; padding: 18px; color: #FFFFFF; font-size: 1.45rem; font-weight: 800;
          text-align: center; margin-bottom: 12px; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18); }}
.banner small {{ font-size: 1rem; font-weight: 600; }}
.enlace {{ display: block; text-align: center; padding: 14px 10px; border-radius: 14px; color: #FFFFFF !important;
          text-decoration: none !important; font-weight: 700; font-size: 1.05rem;
          box-shadow: 0 4px 12px rgba(11, 60, 93, 0.2); }}
.enlace:hover {{ filter: brightness(1.12); }}
.enlace.inactivo {{ background: #66757F; cursor: default; }}
.aviso {{ background: #FFF4E0; border-left: 8px solid {AMBAR}; color: #4A3200; padding: 14px 18px;
         border-radius: 10px; font-size: 1.05rem; }}
button[kind="primary"] {{ background: {TEAL} !important; border-color: {TEAL} !important;
                          font-size: 1.15rem !important; font-weight: 700 !important; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.1rem; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


# ==============================================================================
# 2. FUNCIONES DE APOYO
# ==============================================================================
def titulo_seccion(texto):
    st.markdown(f'<div class="titulo">{texto}</div>', unsafe_allow_html=True)


def tarjeta(etiqueta, valor, color, detalle=""):
    det = f'<div class="det">{detalle}</div>' if detalle else ""
    st.markdown(f'<div class="tarjeta" style="background:{color};"><div class="etq">{etiqueta}</div>'
                f'<div class="val">{valor}</div>{det}</div>', unsafe_allow_html=True)


def tabla_md(encabezados, filas):
    """Tabla en markdown: se lee con claridad y hereda el color del tema."""
    lineas = ["| " + " | ".join(encabezados) + " |",
              "|" + "|".join([":---"] + [":---:"] * (len(encabezados) - 1)) + "|"]
    lineas += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    st.markdown("\n".join(lineas))


def imagen_figura(fig, ancho):
    """Muestra una figura como imagen de ancho fijo: la letra queda proporcional al texto."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.image(buffer, width=ancho)


def figura_centrada(fig, ancho=620):
    _, centro, _ = st.columns([1, 3, 1])
    with centro:
        imagen_figura(fig, ancho)


@st.cache_data
def cargar_metricas():
    with open(ARCHIVO_METRICAS, encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_resource(show_spinner="Cargando la red neuronal (solo la primera vez)...")
def cargar_modelo():
    """La importación de TensorFlow se hace aquí para que la página cargue rápido."""
    import tensorflow as tf
    return tf, tf.keras.models.load_model(ARCHIVO_MODELO)


def analizar(imagen_pil, umbral, tamano):
    """Mismo preprocesamiento del notebook + predicción + Grad-CAM.
    Devuelve (probabilidad de neumonía, imagen 224x224 uint8, mapa de calor 224x224)."""
    tf, modelo = cargar_modelo()
    rgb = np.array(imagen_pil.convert("RGB"))
    imagen = tf.image.resize(rgb, (tamano, tamano))
    imagen = tf.cast(tf.clip_by_value(imagen, 0, 255), tf.uint8)

    capa_escalado = modelo.get_layer("escalado")
    capa_base = modelo.get_layer("mobilenetv2_1.00_224")
    capa_promedio = modelo.get_layer("promedio_global")
    capa_salida = modelo.get_layer("prob_neumonia")

    x = capa_escalado(tf.cast(imagen[None, ...], tf.float32))
    with tf.GradientTape() as cinta:
        mapas = capa_base(x, training=False)
        cinta.watch(mapas)
        prob = capa_salida(capa_promedio(mapas))[0, 0]
        objetivo = prob if prob >= umbral else 1.0 - prob
    gradientes = cinta.gradient(objetivo, mapas)[0]
    pesos = tf.reduce_mean(gradientes, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(mapas[0] * pesos, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (tamano, tamano))[..., 0]
    return float(prob), imagen.numpy(), cam.numpy()


# ==============================================================================
# 3. CARGA DE EVIDENCIAS
# ==============================================================================
if not os.path.exists(ARCHIVO_METRICAS):
    st.error(f"No se encontró '{ARCHIVO_METRICAS}'. Súbalo al repositorio junto a app.py.")
    st.stop()

ev = cargar_metricas()
UMBRAL = ev["umbral"]
TAMANO = ev["tamano_imagen"]
CLAVE_SEL = next(k for k in ev["metricas_prueba"] if "seleccionado" in k)
CLAVE_REF = next(k for k in ev["metricas_prueba"] if "referencia" in k)
TEST = ev["metricas_prueba"][CLAVE_SEL]
TEST_REF = ev["metricas_prueba"][CLAVE_REF]

# ==============================================================================
# 4. ENCABEZADO, AVISO Y ENLACES
# ==============================================================================
st.markdown(f"""
<div style="background: linear-gradient(90deg,{MARINO},{AZUL});
            padding:25px; border-radius:12px; color:white; text-align:center;">
  <h1 style="color:white; margin-bottom:0; font-size:2.8rem;">⚕️ NeumoScan AI</h1>
  <h3 style="color:white; margin-top:4px;">Tamizaje de neumonía en radiografías de tórax pediátricas
      con redes neuronales convolucionales</h3>
  <p style="font-size:1.05rem; margin-bottom:0; line-height:1.6;">
    Especialización en Inteligencia Artificial – CUN · Proyecto ACA 2 – Aprendizaje Automático<br>
    <b>Curso:</b> APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4<br>
    <b>Autora:</b> Claudia Teresa Aranguren Peña<br>
    <b>Docente:</b> Dr. Hamilton Rivera Flor
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="aviso">📋 <b>Herramienta académica de tamizaje.</b> Estima la probabilidad de neumonía '
    'en radiografías de tórax frontales de niños de 1 a 5 años, para priorizar la revisión médica. '
    '<b>No reemplaza el diagnóstico de un profesional de la salud.</b></div>',
    unsafe_allow_html=True,
)

titulo_seccion("📌 Acceso al proyecto")
for columna, color, (nombre, url) in zip(st.columns(len(ENLACES)), [AZUL, MARINO, TEAL, PURPURA], ENLACES.items()):
    with columna:
        if url:
            st.markdown(f'<a class="enlace" style="background:{color};" href="{url}" target="_blank">'
                        f'{nombre}</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="enlace inactivo">{nombre}<br><small>Disponible pronto</small></span>',
                        unsafe_allow_html=True)

# ==============================================================================
# 5. ANALIZAR UNA RADIOGRAFÍA
# ==============================================================================
titulo_seccion("🔍 Analizar una radiografía")
col_entrada, col_resultado = st.columns([1, 1.25], gap="large")

with col_entrada:
    fuente = st.radio("¿Qué radiografía quiere analizar?",
                      ["Usar una radiografía de ejemplo", "Subir mi propia radiografía"])
    imagen_pil = None
    clave_imagen = None
    if fuente == "Subir mi propia radiografía":
        archivo = st.file_uploader("Radiografía de tórax frontal (JPG o PNG)", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            imagen_pil = Image.open(archivo)
            clave_imagen = f"subida:{archivo.name}:{archivo.size}"
    else:
        ejemplos = sorted(f for f in os.listdir(CARPETA_EJEMPLOS) if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
            if os.path.isdir(CARPETA_EJEMPLOS) else []
        if ejemplos:
            nombres = {f"{'Normal' if f.startswith('normal') else 'Neumonía'} · ejemplo "
                       f"{os.path.splitext(f)[0].split('_')[-1]}": f for f in ejemplos}
            elegido = st.selectbox("Ejemplo (radiografías del conjunto de prueba)", list(nombres))
            imagen_pil = Image.open(os.path.join(CARPETA_EJEMPLOS, nombres[elegido]))
            clave_imagen = f"ejemplo:{nombres[elegido]}"
        else:
            st.warning("No hay radiografías de ejemplo en la carpeta 'ejemplos'.")

    # Si cambia la radiografía, se borra el resultado anterior para no mostrarlo con otra imagen
    if st.session_state.get("clave_imagen") != clave_imagen:
        st.session_state["clave_imagen"] = clave_imagen
        st.session_state.pop("resultado", None)

    if imagen_pil is not None:
        st.image(imagen_pil.convert("L"), caption="Radiografía seleccionada", width=330)
        if st.button("Analizar radiografía", type="primary"):
            with st.spinner("La red está analizando la radiografía..."):
                st.session_state["resultado"] = analizar(imagen_pil, UMBRAL, TAMANO)

with col_resultado:
    resultado = st.session_state.get("resultado")
    if imagen_pil is None or resultado is None:
        st.info("👈 Elija una radiografía y pulse **Analizar radiografía**.")
    else:
        prob, imagen_224, calor = resultado
        positivo = prob >= UMBRAL
        if positivo:
            st.markdown(f'<div class="banner" style="background:{CORAL};">⚠️ Tamizaje POSITIVO<br>'
                        f'<small>Hallazgos compatibles con neumonía: priorizar revisión médica</small></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner" style="background:{VERDE};">✅ Tamizaje NEGATIVO<br>'
                        f'<small>Sin hallazgos compatibles con neumonía según el modelo</small></div>',
                        unsafe_allow_html=True)
        tarjeta("Probabilidad estimada de neumonía", f"{prob * 100:.1f}%",
                CORAL if positivo else VERDE, f"Umbral de decisión: {UMBRAL * 100:.0f}%")
        st.progress(min(max(prob, 0.0), 1.0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.2))
        ax1.imshow(imagen_224)
        ax1.set_title("Radiografía (224 × 224)")
        ax2.imshow(imagen_224)
        ax2.imshow(calor, cmap="jet", alpha=0.45)
        ax2.set_title("¿Dónde miró la red? (Grad-CAM)")
        for ax in (ax1, ax2):
            ax.axis("off")
        imagen_figura(fig, 560)
        st.markdown(
            f"**Cómo leerlo:** rojo y amarillo marcan las zonas que más pesaron en la decisión. "
            f"El umbral es {UMBRAL:.2f} (y no 0.50) para priorizar la detección de enfermos: en validación "
            f"detectó al menos el {ev['recall_objetivo_validacion'] * 100:.0f} % de los casos de neumonía."
        )
        st.markdown("**Recomendación:** " + (
            "valoración médica prioritaria y confirmación diagnóstica." if positivo else
            "control según criterio clínico. Un resultado negativo no descarta neumonía si hay síntomas."))
        st.caption("El modelo responde para cualquier imagen; solo es válido con radiografías de tórax "
                   "frontales pediátricas como las del entrenamiento.")

# ==============================================================================
# 6. CONOCE LOS DATOS
# ==============================================================================
titulo_seccion("📊 Conoce los datos")
part = pd.DataFrame(ev["particiones"]).set_index("particion").reindex(["entrenamiento", "validacion", "prueba"])
total_img = int(part["imagenes"].sum())
total_neu = int(part["neumonia"].sum())
st.markdown(
    f"**Chest X-Ray Images (Pneumonia)** (Kermany et al., 2018; Mooney, 2018): **{total_img:,} radiografías** "
    f"de tórax de niños de 1 a 5 años, gestionadas en una base de datos **SQLite**. "
    f"Se dividieron **por paciente** para que ninguna persona quedara a la vez en entrenamiento y prueba."
    .replace(",", "."))

g1, g2 = st.columns([1, 1.1])
with g1:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    valores = [total_img - total_neu, total_neu]
    barras = ax.bar(["Normal", "Neumonía"], valores, color=[COLOR_NORMAL, COLOR_NEUMONIA])
    ax.bar_label(barras, labels=[f"{v:,} ({v / total_img:.0%})".replace(",", ".") for v in valores])
    ax.set_ylim(0, max(valores) * 1.18)
    ax.set_ylabel("Radiografías")
    ax.set_title("Clases desbalanceadas")
    imagen_figura(fig, 440)
with g2:
    tabla_md(["Grupo", "Pacientes", "Radiografías", "Normal", "Neumonía"], [
        [f"**{nombre.capitalize().replace('Validacion', 'Validación')}**", int(f["pacientes"]),
         int(f["imagenes"]), int(f["normal"]), int(f["neumonia"])]
        for nombre, f in part.iterrows()])
    st.markdown(
        "El desbalance (≈ 73 % neumonía) se compensó con **pesos por clase**, y el modelo se evalúa con "
        "Recall, especificidad, F1 y ROC-AUC, no solo con exactitud. Las imágenes se llevaron a "
        "224 × 224 y en el entrenamiento se aplicó **aumento de datos** (giros, zoom, desplazamiento y "
        "contraste leves, sin volteo para respetar la anatomía).")

# ==============================================================================
# 7. DESEMPEÑO DEL MODELO
# ==============================================================================
titulo_seccion("📈 Desempeño del modelo")
st.markdown(f"**Modelo:** {ev['modelo']} · **Evaluación única** en el conjunto de prueba "
            f"({int(part.loc['prueba', 'imagenes'])} radiografías de {int(part.loc['prueba', 'pacientes'])} "
            f"pacientes nunca vistos) con el umbral {UMBRAL}.")

columnas = st.columns(5)
for col, (clave, etiqueta, color) in zip(columnas, [
        ("Recall (sensibilidad)", "Recall", TEAL), ("Especificidad", "Especificidad", AZUL),
        ("Precision", "Precisión", PURPURA), ("F1-Score", "F1-Score", VERDE), ("ROC-AUC", "ROC-AUC", MARINO)]):
    with col:
        tarjeta(etiqueta, f"{TEST[clave]:.3f}", color)

fn, fp = int(TEST["Falsos negativos"]), int(TEST["Falsos positivos"])
st.markdown(f"De cada 100 niños con neumonía, el modelo detecta unos **{TEST['Recall (sensibilidad)'] * 100:.0f}**. "
            f"Con el umbral {UMBRAL}, los enfermos no detectados bajaron de "
            f"**{int(TEST_REF['Falsos negativos'])} a {fn}** frente al umbral 0.5, a cambio de "
            f"**{fp - int(TEST_REF['Falsos positivos'])}** falsas alarmas adicionales.")

tab_curvas, tab_matriz, tab_roc, tab_umbral, tab_gradcam = st.tabs(
    ["📉 Curvas de aprendizaje", "🎯 Matriz de confusión", "📈 Curva ROC", "⚖️ Umbral", "🔥 Grad-CAM"])

with tab_curvas:
    h = ev["historial"]
    epocas = range(1, len(h["loss"]) + 1)
    fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (clave, titulo) in zip(ejes, [("loss", "Pérdida (loss)"), ("accuracy", "Exactitud"), ("auc", "ROC-AUC")]):
        ax.plot(epocas, h[clave], "o-", color=COLOR_NORMAL, label="Entrenamiento")
        ax.plot(epocas, h["val_" + clave], "s-", color=COLOR_NEUMONIA, label="Validación")
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    imagen_figura(fig, 980)
    st.markdown("Las curvas de entrenamiento y validación avanzan juntas: el modelo **generaliza** y no "
                "memoriza. El entrenamiento se detuvo solo (*early stopping*) cuando el AUC de validación "
                "dejó de mejorar. Un *fine-tuning* adicional se probó y se descartó porque produjo sobreajuste.")

with tab_matriz:
    (vn, fp_m), (fn_m, vp) = ev["matriz_confusion"]
    tabla_md(["", "Predicho: Normal", "Predicho: Neumonía"],
             [["**Real: Normal**", vn, fp_m], ["**Real: Neumonía**", fn_m, vp]])
    m1, m2, m3, m4 = st.columns(4)
    for col, etiqueta, valor, color in [(m1, "Verdaderos negativos", vn, AZUL), (m2, "Falsos positivos", fp_m, AMBAR),
                                        (m3, "Falsos negativos", fn_m, CORAL), (m4, "Verdaderos positivos", vp, VERDE)]:
        with col:
            tarjeta(etiqueta, valor, color)

with tab_roc:
    roc = ev["roc"]
    (vn, fp_m), (fn_m, vp) = ev["matriz_confusion"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(roc["fpr"], roc["tpr"], linewidth=2, color=COLOR_NORMAL, label=f"CNN (AUC = {roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
    ax.scatter(fp_m / (fp_m + vn), vp / (vp + fn_m), color=COLOR_NEUMONIA, s=60, zorder=3,
               label=f"Punto de operación (umbral {UMBRAL})")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title("Curva ROC en el conjunto de prueba")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    figura_centrada(fig, 560)

with tab_umbral:
    st.markdown(f"Regla fijada **antes** de ver la prueba: en validación, el umbral más alto con "
                f"Recall ≥ {ev['recall_objetivo_validacion']:.2f}. Resultado: **{UMBRAL}**.")
    tabla_md(["Umbral", "Recall", "Especificidad", "Precisión", "Falsos negativos", "Falsos positivos"], [
        [f"**{f['umbral']:.2f}** ✅" if abs(f["umbral"] - UMBRAL) < 1e-9 else f"{f['umbral']:.2f}",
         f"{f['recall']:.3f}", f"{f['especificidad']:.3f}", f"{f['precision']:.3f}",
         int(f["falsos_negativos"]), int(f["falsos_positivos"])]
        for f in ev["tabla_umbrales_validacion"] if 0.2 <= f["umbral"] <= 0.6])

with tab_gradcam:
    if os.path.exists(ARCHIVO_GRADCAM):
        st.image(ARCHIVO_GRADCAM)
    st.markdown("En los casos de **neumonía**, la red se fija sobre todo en los **pulmones**. En algunos casos "
                "**normales** mira zonas fuera de ellos (hombros, marcador «R», abdomen): una posible señal de "
                "atajos por el encuadre de las imágenes, que se reporta como limitación.")

# ==============================================================================
# 8. METODOLOGÍA Y LIMITACIONES
# ==============================================================================
st.divider()
with st.expander("ℹ️ Arquitectura, metodología, limitaciones y referencias"):
    st.markdown(f"""
**Arquitectura:** entrada 224 × 224 × 3 → aumento de datos (solo al entrenar) → escalado a [−1, 1] →
**MobileNetV2** preentrenada en ImageNet, congelada (extractor de características, 7 × 7 × 1280) →
*Global Average Pooling* → *Dropout* 30 % → neurona sigmoide (probabilidad de neumonía).
Se entrenan solo 1 281 de 2.26 millones de parámetros (*transfer learning*).

**Flujo:** inventario de radiografías en SQLite y EDA con SQL → división 70/15/15 por paciente →
pipeline `tf.data` y aumento de datos → entrenamiento con pesos por clase y *early stopping* →
umbral elegido en validación → evaluación única en prueba → Grad-CAM.

**Limitaciones:** dataset pediátrico de un solo centro (Guangzhou), métricas por imagen, posible
aprendizaje por atajos detectado con Grad-CAM y sin detección de imágenes ajenas al dominio.
Herramienta académica: no reemplaza el criterio médico.

**TensorFlow del modelo:** {ev['version_tensorflow']}.

**Referencias (APA 7)**

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., ... Zhang, K. (2018).
Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5),
1122–1131. https://doi.org/10.1016/j.cell.2018.02.010

Mooney, P. (2018). *Chest X-Ray Images (Pneumonia)* [Conjunto de datos]. Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals
and linear bottlenecks. En *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition* (pp. 4510–4520).
""")
