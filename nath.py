import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import warnings
from gtts import gTTS
import base64

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

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'modelo_entrenado.h5')
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Modelo está cargando..'):
    model = load_model()

# Generar saludo
def generar_saludo():
    texto = "¡Hola! soy Órasi, tu asistente neuronal personal, ¿Que producto vamos a identificar hoy?"
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    try:
        audio_bytes = mp3_fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al reproducir el audio: {e}")

# Reproducir el saludo al inicio
mp3_fp = generar_saludo()
reproducir_audio(mp3_fp)

# Título de la página
st.image("./videos/banner.png", use_column_width=True)
st.write("# Detección de Productos")

def import_and_predict(image_data, model, class_names):
    if image_data.mode != 'RGB':
        image_data = image_data.convert('RGB')
        
    image_data = image_data.resize((224, 224))  # Ajustado a 224x224
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Crear un batch
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index].strip()
    return class_name, score

def generar_audio(texto):
    tts = gTTS(text=texto, lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

class_names = open("./clases (1).txt", "r").readlines()

option = st.selectbox(
    "¿Qué te gustaría usar para subir la foto?",
    ("Tomar foto", "Subir archivo", "URL"),
    index=None,
    placeholder="Selecciona cómo subir la foto"
)
confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

img_file_buffer = None

if option == "Tomar foto":
    img_file_buffer = st.camera_input("Capture una foto para identificar el producto")
elif option == "Subir archivo":
    img_file_buffer = st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])
elif option == "URL":
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Procesar la imagen y realizar la predicción
if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)

        # Realizar la predicción
        class_name, score = import_and_predict(image, model, class_names)
        max_score = np.max(score)

        # Mostrar el resultado y generar audio
        if max_score > confianza:
            resultado = f"Tipo de Producto: {class_name}\nPuntuación de confianza: {100 * max_score:.2f}%"
            st.subheader(f"Tipo de Producto: {class_name}")
            st.text(f"Puntuación de confianza: {100 * max_score:.2f}%")
        else:
            resultado = "No se pudo determinar el tipo de producto"
            st.text(resultado)

        # Generar y reproducir el audio
        mp3_fp = generar_audio(resultado)
        reproducir_audio(mp3_fp)
        
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
