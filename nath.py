import os  # Importa el módulo os para manejar operaciones del sistema operativo

# Desactiva las optimizaciones OneDNN de TensorFlow para evitar posibles errores o comportamientos inesperados
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st  # Importa Streamlit para crear aplicaciones web interactivas
import tensorflow as tf  # Importa TensorFlow para el modelo de redes neuronales
from tensorflow.keras.applications.vgg16 import preprocess_input  # Importa la función de preprocesamiento de imágenes de VGG16
from PIL import Image  # Importa la librería PIL para manejar imágenes
import numpy as np  # Importa NumPy para operaciones matemáticas y manejo de matrices
import requests  # Importa requests para hacer peticiones HTTP y obtener imágenes desde URLs
from io import BytesIO  # Importa BytesIO para manejar flujos de datos binarios en memoria
import warnings  # Importa warnings para gestionar advertencias del sistema
from gtts import gTTS  # Importa gTTS para generar audio a partir de texto
import base64  # Importa base64 para codificar y decodificar datos en formato base64

# Ignora las advertencias para evitar mensajes innecesarios en la ejecución del programa
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="¿Qué producto es?",
    page_icon="icono.ico",
    initial_sidebar_state='auto',
    menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "Nathalia Quintero & Angelly Cristancho. Inteligencia Artificial *Ejemplo de clase* Ingeniería de sistemas!"
    }
)

# Define un estilo personalizado para ocultar elementos innecesarios de Streamlit
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* Oculta el menú principal de Streamlit */
    footer {visibility: hidden;}  /* Oculta el pie de página */
    .stButton>button {
        background-color: #4CAF50;  /* Establece el color de fondo de los botones */
        color: white;  /* Establece el color del texto de los botones */
        padding: 10px 24px;  /* Define el espaciado interno de los botones */
        border-radius: 8px;  /* Define los bordes redondeados de los botones */
        border: none;  /* Elimina el borde de los botones */
        cursor: pointer;  /* Cambia el cursor al pasar sobre el botón */
    }
    .stButton>button:hover {
        background-color: #45a049;  /* Cambia el color de fondo al pasar el cursor */
    }
    </style>
"""

# Aplica el estilo personalizado a la página de Streamlit
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ------------------ Funciones de Audio (al estilo Nathalia) ------------------

def generar_saludo():
    """Genera un saludo al inicio, al estilo de Nathalia."""
    texto = "¡Hola! soy Cámapp, tu asistente neuronal personal, ¿Qué objeto del laboratorio vamos a identificar hoy?"
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def generar_audio(texto):
    """Genera audio a partir del texto proporcionado."""
    if not texto.strip():
        texto = "No se encontró información para este objeto."
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    """Reproduce el audio generado en Streamlit."""
    audio_bytes = mp3_fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Reproduce el saludo inicial (al estilo Nathalia)
mp3_fp_saludo = generar_saludo()
reproducir_audio(mp3_fp_saludo)

# -----------------------------------------------------------------------------

# Define una función para cargar el modelo de inteligencia artificial con caché para optimizar el rendimiento
@st.cache_resource
def load_model():
    model_path = "./proyectoia.h5"  # Ruta del modelo entrenado
    
    # Verifica si el archivo del modelo existe en la ruta especificada
    if not os.path.exists(model_path):
        st.error("Error: No se encontró el modelo entrenado. Verifica la ruta.")
        return None
    try:
        # Carga el modelo sin compilar para evitar posibles errores
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Muestra un mensaje de carga mientras se ejecuta la función de carga del modelo
with st.spinner('Cargando modelo...'):
    model = load_model()

# Cargar nombres de clases desde un archivo externo
class_names = []
try:
    with open("clase.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip().lower() for line in f.readlines()]
    if not class_names:
        st.error("El archivo clase.txt está vacío.")
except FileNotFoundError:
    st.error("No se encontró el archivo clase.txt.")

# Cargar descripciones de objetos desde un archivo externo
descripcion_dict = {}
try:
    with open("proma.txt", "r", encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split(":", 1)
            if len(partes) == 2:
                clave = partes[0].strip().lower()
                descripcion = partes[1].strip()
                descripcion_dict[clave] = descripcion
except FileNotFoundError:
    st.error("No se encontró el archivo proma.txt.")


# Título de la página
st.image("./videos/banner.png", use_column_width=True)
st.write("# Detección de Productos Cámapp")
st.title("Smart Regions Center")
st.write("Desarrolado por Angelly y Nathalia")
confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

            
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def import_and_predict(image, model, class_names):
    if model is None:
        return "Modelo no cargado", 0.0
    image = preprocess_image(image)
    prediction = model.predict(image)
    index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Desconocido"
    return class_name, confidence

# Captura una imagen desde la cámara o permite la carga de un archivo
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto") or \
                  st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

resultado = "No se ha procesado ninguna imagen."

# Si no hay imagen cargada, permite ingresar una URL
if img_file_buffer is None:
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Si hay una imagen cargada y el modelo está disponible
if img_file_buffer and model:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)
        class_name, confidence_score = import_and_predict(image, model, class_names)
        descripcion = descripcion_dict.get(class_name, "No hay información disponible para este objeto.")
        
        if confidence_score > confianza:
            resultado = f"Objeto Detectado: {class_name.capitalize()}\nConfianza: {100 * confidence_score:.2f}%\nDescripción: {descripcion}"
            st.subheader(f"Tipo de Objeto: {class_name.capitalize()}")
            st.text(f"Confianza: {100 * confidence_score:.2f}%")
            st.write(f"Descripción: {descripcion}")
        else:
            resultado = "No se pudo determinar el tipo de objeto"
            st.text(resultado)
        
        # Limpia el resultado para evitar caracteres especiales
        resultado_limpio = resultado.replace('*', '').replace('_', '').replace('/', '')
        # Genera y reproduce el audio con el resultado
        mp3_fp_resultado = generar_audio(resultado_limpio)
        reproducir_audio(mp3_fp_resultado)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.text("Por favor, cargue una imagen usando una de las opciones anteriores.")

#informacion para tomar foto

with st.expander("Como tomar la FOTO correctamente"):
   
    st.markdown("¿Cómo poner el producto correctamente en la cámara?") 

    # Ruta del archivo de video
    video_file_path = './videos/SI.mp4'
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")

    # Ruta del archivo de video
    video_file_path = './videos/NO.mp4'
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")
