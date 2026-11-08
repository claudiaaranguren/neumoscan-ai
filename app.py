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
  - .streamlit/config.toml: tema de colores (azul brillante) de la interfaz.
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

# Paleta propia de NeumoScan: azul eléctrico y cian sobre fondo claro (distinta al ACA 1)
NOCHE, REAL, AZUL, CIAN = "#0A1F5C", "#1238C9", "#1565FF", "#00B4D8"
INDIGO, TEAL, VERDE = "#4F46E5", "#0E9F8E", "#12A150"
CORAL, AMBAR = "#E5383B", "#E08A00"
TINTA, GRIS = "#0F1E3D", "#5B6B86"
COLOR_NORMAL, COLOR_NEUMONIA = AZUL, CORAL

# Pulmones dibujados en SVG (se ven igual en cualquier sistema operativo, sin emojis)
ICONO_PULMONES = """
<svg viewBox="0 0 64 64" width="66" height="66" fill="none" stroke="white" stroke-width="3.2"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M32 8v20"/><path d="M32 28c-3 0-6 3-8 6"/><path d="M32 28c3 0 6 3 8 6"/>
  <path d="M24 18c-8 2-14 14-14 28 0 6 3 9 8 9 5 0 8-3 8-9V22"/>
  <path d="M40 18c8 2 14 14 14 28 0 6-3 9-8 9-5 0-8-3-8-9V22"/>
</svg>"""

st.markdown(f"""
<style>
html {{ font-size: 18px; }}

/* Fondo de la página: azul muy claro con un resplandor suave arriba */
.stApp {{
  background:
    radial-gradient(900px 420px at 0% 0%, #D6E6FF 0%, rgba(214,230,255,0) 70%),
    radial-gradient(900px 420px at 100% 0%, #D3F4FA 0%, rgba(211,244,250,0) 70%),
    #F2F6FD;
}}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Encabezado principal */
.hero {{ position: relative; overflow: hidden; border-radius: 26px; padding: 34px 38px 26px;
        background: linear-gradient(120deg, {NOCHE} 0%, {REAL} 38%, {AZUL} 68%, {CIAN} 100%);
        color: #FFFFFF; box-shadow: 0 18px 40px rgba(21, 101, 255, 0.28); }}
.hero::before, .hero::after {{ content: ""; position: absolute; border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.16); }}
.hero::before {{ width: 420px; height: 420px; right: -120px; top: -190px; }}
.hero::after  {{ width: 360px; height: 360px; left: -140px; bottom: -220px; }}
.hero-top {{ display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px;
             position: relative; }}
.logo {{ width: 104px; height: 104px; border-radius: 28px; box-shadow: 0 0 0 8px rgba(255,255,255,0.08), 0 10px 26px rgba(0,0,0,0.18); background: rgba(255,255,255,0.16);
        border: 2px solid rgba(255,255,255,0.35); display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; }}
.kicker {{ font-size: 0.95rem; font-weight: 800; letter-spacing: 0.22em; margin-top: 6px; text-transform: uppercase;
          color: #BFF3FF; }}
.hero h1 {{ color: #FFFFFF !important; font-size: 4.2rem; font-weight: 900; margin: 0; padding: 0;
           line-height: 1.05; letter-spacing: -0.01em; text-shadow: 0 6px 24px rgba(0, 20, 80, 0.35); }}
.hero h1 span {{ color: #7CE7FF; }}
.hero .sub {{ font-size: 1.5rem; font-weight: 700; margin: 2px 0 0; color: #EAF2FF; }}
.hero .raya {{ width: 120px; height: 5px; border-radius: 5px; margin: 10px auto 0;
              background: linear-gradient(90deg, #7CE7FF, #FFFFFF); }}
.chips {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 22px; position: relative; }}
.chip {{ background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.35); border-radius: 999px;
        padding: 6px 14px; font-size: 0.92rem; }}
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 22px; position: relative; }}
.stat {{ background: #FFFFFF; border-radius: 16px; padding: 12px 14px; text-align: center; color: {TINTA};
        box-shadow: 0 6px 16px rgba(10, 31, 92, 0.18); }}
.stat .n {{ font-size: 1.75rem; font-weight: 900; color: {AZUL}; line-height: 1.1; }}
.stat .t {{ font-size: 0.85rem; font-weight: 700; color: {GRIS}; text-transform: uppercase; letter-spacing: 0.04em; }}
@media (max-width: 900px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} .hero h1 {{ font-size: 2.6rem; }} }}

/* Títulos de sección: número + texto + línea degradada */
.sec {{ display: flex; align-items: center; gap: 14px; margin: 34px 0 14px; }}
.sec .num {{ background: linear-gradient(135deg, {AZUL}, {CIAN}); color: #FFFFFF; font-weight: 900;
            font-size: 1.15rem; width: 48px; height: 48px; border-radius: 14px; display: flex;
            align-items: center; justify-content: center; box-shadow: 0 6px 14px rgba(21, 101, 255, 0.35); }}
.sec .txt {{ font-size: 1.7rem; font-weight: 900; color: {NOCHE}; }}
.sec .linea {{ flex: 1; height: 4px; border-radius: 4px;
              background: linear-gradient(90deg, {AZUL}, {CIAN} 40%, rgba(0,180,216,0)); }}

/* Contenedores con clave "tarjeta_..." = tarjetas blancas */
div[class*="st-key-tarjeta"] {{
  background: #FFFFFF; border-radius: 22px !important; border: 1px solid #DCE6F7 !important;
  box-shadow: 0 10px 28px rgba(16, 60, 140, 0.08); }}
div[class*="st-key-tarjeta"] {{ padding: 24px 26px; }}

/* Indicadores (KPI): fondo teñido + borde superior de color */
.kpi {{ border-radius: 18px; padding: 16px 12px 14px; text-align: center; margin-bottom: 12px;
       border-top: 7px solid var(--c); background: var(--f); }}
.kpi .etq {{ font-size: 0.88rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: {GRIS}; }}
.kpi .val {{ font-size: 2.5rem; font-weight: 900; line-height: 1.15; margin-top: 4px; color: var(--c); }}
.kpi .det {{ font-size: 0.9rem; color: {TINTA}; font-weight: 600; }}
.kpi.solido {{ background: var(--c); border-top-color: rgba(255,255,255,0.35); }}
.kpi.solido .etq, .kpi.solido .val, .kpi.solido .det {{ color: #FFFFFF; }}

.banner {{ border-radius: 20px; padding: 20px; color: #FFFFFF; font-size: 1.55rem; font-weight: 900;
          text-align: center; margin-bottom: 12px; box-shadow: 0 10px 24px rgba(15, 30, 61, 0.18); }}
.banner small {{ font-size: 1rem; font-weight: 600; }}

/* Botones de enlaces */
.enlace {{ display: block; text-align: center; padding: 15px 10px; border-radius: 999px; color: #FFFFFF !important;
          text-decoration: none !important; font-weight: 800; font-size: 1.02rem;
          box-shadow: 0 8px 18px rgba(21, 101, 255, 0.25); transition: transform .15s; }}
.enlace:hover {{ transform: translateY(-2px); filter: brightness(1.08); }}
.enlace.inactivo {{ background: #FFFFFF; color: {GRIS} !important; border: 2px dashed #B8C6DD;
                   box-shadow: none; cursor: default; padding: 9px 10px; }}
.enlace.inactivo:hover {{ transform: none; }}
.enlace small {{ font-weight: 600; }}

.aviso {{ background: #FFF7E8; border: 1px solid #FFD999; border-left: 8px solid {AMBAR}; color: #4A3200;
         padding: 14px 18px; border-radius: 14px; font-size: 1.05rem; margin-top: 18px; }}
.pie {{ text-align: center; color: {GRIS}; font-size: 0.9rem; margin: 30px 0 10px; }}

button[kind="primary"] {{ background: linear-gradient(90deg, {REAL}, {AZUL}) !important; border: none !important;
                          border-radius: 999px !important; font-size: 1.15rem !important; font-weight: 800 !important;
                          padding: 0.55rem 1.6rem !important; box-shadow: 0 8px 18px rgba(21, 101, 255, 0.3); }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.1rem; font-weight: 800; }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": NOCHE, "axes.edgecolor": "#9AA9C2",
                     "axes.spines.top": False, "axes.spines.right": False})


# ==============================================================================
# 2. FUNCIONES DE APOYO
# ==============================================================================
def titulo_seccion(numero, texto):
    st.markdown(f'<div class="sec"><span class="num">{numero:02d}</span><span class="txt">{texto}</span>'
                f'<span class="linea"></span></div>', unsafe_allow_html=True)


def tinte(color_hex, alfa=0.09):
    """Versión transparente de un color, para el fondo de los indicadores."""
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alfa})"


def tarjeta(etiqueta, valor, color, detalle="", solido=False):
    det = f'<div class="det">{detalle}</div>' if detalle else ""
    clase = "kpi solido" if solido else "kpi"
    st.markdown(f'<div class="{clase}" style="--c:{color}; --f:{tinte(color)};"><div class="etq">{etiqueta}</div>'
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
part = pd.DataFrame(ev["particiones"]).set_index("particion").reindex(["entrenamiento", "validacion", "prueba"])
total_img = int(part["imagenes"].sum())
total_neu = int(part["neumonia"].sum())
total_txt = f"{total_img:,}".replace(",", ".")

# ==============================================================================
# 4. ENCABEZADO, AVISO Y ENLACES
# ==============================================================================
st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <div class="logo">{ICONO_PULMONES}</div>
    <div>
      <div class="kicker">Proyecto ACA 2 · Redes neuronales convolucionales</div>
      <h1>NeumoScan <span>AI</span></h1>
      <p class="sub">Tamizaje de neumonía en radiografías de tórax pediátricas</p>
      <div class="raya"></div>
    </div>
  </div>
  <div class="chips">
    <span class="chip"><b>Autora:</b> Claudia Teresa Aranguren Peña</span>
    <span class="chip"><b>Docente:</b> Dr. Hamilton Rivera Flor</span>
    <span class="chip"><b>Curso:</b> APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4</span>
    <span class="chip">Especialización en Inteligencia Artificial – CUN</span>
  </div>
  <div class="stats">
    <div class="stat"><div class="n">{total_txt}</div><div class="t">Radiografías</div></div>
    <div class="stat"><div class="n">{TEST['Recall (sensibilidad)'] * 100:.1f} %</div><div class="t">Neumonías detectadas</div></div>
    <div class="stat"><div class="n">{TEST['ROC-AUC']:.3f}</div><div class="t">ROC-AUC en prueba</div></div>
    <div class="stat"><div class="n">MobileNetV2</div><div class="t">CNN · transfer learning</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="aviso">📋 <b>Herramienta académica de tamizaje.</b> Estima la probabilidad de neumonía '
    'en radiografías de tórax frontales de niños de 1 a 5 años, para priorizar la revisión médica. '
    '<b>No reemplaza el diagnóstico de un profesional de la salud.</b></div>',
    unsafe_allow_html=True,
)

titulo_seccion(1, "Acceso al proyecto")
colores_enlace = [f"linear-gradient(90deg,{REAL},{AZUL})", f"linear-gradient(90deg,{NOCHE},{REAL})",
                  f"linear-gradient(90deg,{AZUL},{CIAN})", f"linear-gradient(90deg,{INDIGO},{AZUL})"]
for columna, fondo, (nombre, url) in zip(st.columns(len(ENLACES)), colores_enlace, ENLACES.items()):
    with columna:
        if url:
            st.markdown(f'<a class="enlace" style="background:{fondo};" href="{url}" target="_blank">'
                        f'{nombre}</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="enlace inactivo">{nombre}<br><small>Disponible pronto</small></span>',
                        unsafe_allow_html=True)

# ==============================================================================
# 5. ANALIZAR UNA RADIOGRAFÍA
# ==============================================================================
titulo_seccion(2, "Analizar una radiografía")
with st.container(key="tarjeta_analisis"):
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
            ejemplos = sorted(f for f in os.listdir(CARPETA_EJEMPLOS)
                              if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
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
                st.markdown(f'<div class="banner" style="background:linear-gradient(120deg,#B4161B,{CORAL});">'
                            f'⚠️ Tamizaje POSITIVO<br><small>Hallazgos compatibles con neumonía: '
                            f'priorizar revisión médica</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="banner" style="background:linear-gradient(120deg,#0B7A3B,{VERDE});">'
                            f'✅ Tamizaje NEGATIVO<br><small>Sin hallazgos compatibles con neumonía '
                            f'según el modelo</small></div>', unsafe_allow_html=True)
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
titulo_seccion(3, "Conoce los datos")
with st.container(key="tarjeta_datos"):
    st.markdown(
        f"**Chest X-Ray Images (Pneumonia)** (Kermany et al., 2018; Mooney, 2018): **{total_txt} radiografías** "
        f"de tórax de niños de 1 a 5 años, gestionadas en una base de datos **SQLite**. "
        f"Se dividieron **por paciente** para que ninguna persona quedara a la vez en entrenamiento y prueba.")

    g1, g2 = st.columns([1, 1.1])
    with g1:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        valores = [total_img - total_neu, total_neu]
        barras = ax.bar(["Normal", "Neumonía"], valores, color=[COLOR_NORMAL, COLOR_NEUMONIA], width=0.6)
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
titulo_seccion(4, "Desempeño del modelo")
with st.container(key="tarjeta_desempeno"):
    st.markdown(f"**Modelo:** {ev['modelo']} · **Evaluación única** en el conjunto de prueba "
                f"({int(part.loc['prueba', 'imagenes'])} radiografías de {int(part.loc['prueba', 'pacientes'])} "
                f"pacientes nunca vistos) con el umbral {UMBRAL}.")

    columnas = st.columns(5)
    for col, (clave, etiqueta, color) in zip(columnas, [
            ("Recall (sensibilidad)", "Recall", TEAL), ("Especificidad", "Especificidad", AZUL),
            ("Precision", "Precisión", INDIGO), ("F1-Score", "F1-Score", VERDE), ("ROC-AUC", "ROC-AUC", NOCHE)]):
        with col:
            tarjeta(etiqueta, f"{TEST[clave]:.3f}", color)

    fn, fp = int(TEST["Falsos negativos"]), int(TEST["Falsos positivos"])
    st.markdown(f"De cada 100 niños con neumonía, el modelo detecta unos "
                f"**{TEST['Recall (sensibilidad)'] * 100:.0f}**. Con el umbral {UMBRAL}, los enfermos no "
                f"detectados bajaron de **{int(TEST_REF['Falsos negativos'])} a {fn}** frente al umbral 0.5, "
                f"a cambio de **{fp - int(TEST_REF['Falsos positivos'])}** falsas alarmas adicionales.")

    tab_curvas, tab_matriz, tab_roc, tab_umbral, tab_gradcam = st.tabs(
        ["📉 Curvas de aprendizaje", "🎯 Matriz de confusión", "📈 Curva ROC", "⚖️ Umbral", "🔥 Grad-CAM"])

    with tab_curvas:
        h = ev["historial"]
        epocas = range(1, len(h["loss"]) + 1)
        fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, (clave, titulo) in zip(ejes, [("loss", "Pérdida (loss)"), ("accuracy", "Exactitud"),
                                              ("auc", "ROC-AUC")]):
            ax.plot(epocas, h[clave], "o-", color=COLOR_NORMAL, label="Entrenamiento")
            ax.plot(epocas, h["val_" + clave], "s-", color=COLOR_NEUMONIA, label="Validación")
            ax.set_title(titulo)
            ax.set_xlabel("Época")
            ax.grid(alpha=0.3)
            ax.legend()
        imagen_figura(fig, 980)
        st.markdown("Las curvas de entrenamiento y validación avanzan juntas: el modelo **generaliza** y no "
                    "memoriza. El entrenamiento se detuvo solo (*early stopping*) cuando el AUC de validación "
                    "dejó de mejorar. Un *fine-tuning* adicional se probó y se descartó porque produjo "
                    "sobreajuste.")

    with tab_matriz:
        (vn, fp_m), (fn_m, vp) = ev["matriz_confusion"]
        tabla_md(["", "Predicho: Normal", "Predicho: Neumonía"],
                 [["**Real: Normal**", vn, fp_m], ["**Real: Neumonía**", fn_m, vp]])
        m1, m2, m3, m4 = st.columns(4)
        for col, etiqueta, valor, color in [(m1, "Verdaderos negativos", vn, AZUL),
                                            (m2, "Falsos positivos", fp_m, AMBAR),
                                            (m3, "Falsos negativos", fn_m, CORAL),
                                            (m4, "Verdaderos positivos", vp, VERDE)]:
            with col:
                tarjeta(etiqueta, valor, color)

    with tab_roc:
        roc = ev["roc"]
        (vn, fp_m), (fn_m, vp) = ev["matriz_confusion"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.fill_between(roc["fpr"], roc["tpr"], color=COLOR_NORMAL, alpha=0.08)
        ax.plot(roc["fpr"], roc["tpr"], linewidth=2.2, color=COLOR_NORMAL, label=f"CNN (AUC = {roc['auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
        ax.scatter(fp_m / (fp_m + vn), vp / (vp + fn_m), color=COLOR_NEUMONIA, s=70, zorder=3,
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
        st.markdown("En los casos de **neumonía**, la red se fija sobre todo en los **pulmones**. En algunos "
                    "casos **normales** mira zonas fuera de ellos (hombros, marcador «R», abdomen): una posible "
                    "señal de atajos por el encuadre de las imágenes, que se reporta como limitación.")

# ==============================================================================
# 8. METODOLOGÍA Y LIMITACIONES
# ==============================================================================
titulo_seccion(5, "Arquitectura y metodología")
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

st.markdown('<div class="pie">NeumoScan AI · Claudia Teresa Aranguren Peña · Aprendizaje Automático · CUN 2026</div>',
            unsafe_allow_html=True)
