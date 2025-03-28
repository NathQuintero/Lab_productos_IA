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

# Ocultar menú y pie de página de Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), 'modelo_entrenado.h5')
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner('Cargando el modelo...'):
    model = load_model()

# Generar saludo con audio
def generar_saludo():
    texto = "¡Hola! soy Órasi, tu asistente neuronal personal, ¿Qué producto vamos a identificar hoy?"
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

mp3_fp = generar_saludo()
reproducir_audio(mp3_fp)

# Banner y título
st.image("./videos/banner.png", use_column_width=True)
st.write("# Detección de Productos")

def import_and_predict(image_data, model, class_names):
    if image_data.mode != 'RGB':
        image_data = image_data.convert('RGB')
    
    image_data = image_data.resize((224, 224))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  
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

# Sección de carga de imagen
st.sidebar.header("Subir imagen")
option = st.sidebar.radio("Seleccione el método:", ["Tomar foto", "Subir archivo", "URL"], index=1)
confianza = st.sidebar.slider("Nivel de confianza", 0, 100, 50) / 100

img_file_buffer = None
if option == "Tomar foto":
    img_file_buffer = st.camera_input("Capture una foto para identificar el producto")
elif option == "Subir archivo":
    img_file_buffer = st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])
elif option == "URL":
    image_url = st.text_input("Ingrese la URL de la imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Procesamiento de imagen y predicción
if img_file_buffer:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, use_column_width=True)
        
        class_name, score = import_and_predict(image, model, class_names)
        max_score = np.max(score)
        
        if max_score > confianza:
            resultado = f"Tipo de Producto: {class_name}\nPuntuación de confianza: {100 * max_score:.2f}%"
            st.success(f"**Tipo de Producto:** {class_name}")
            st.write(f"**Puntuación de confianza:** {100 * max_score:.2f}%")
        else:
            resultado = "No se pudo determinar el tipo de producto"
            st.warning(resultado)
        
        mp3_fp = generar_audio(resultado)
        reproducir_audio(mp3_fp)
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.info("Por favor, cargue una imagen usando una de las opciones anteriores.")

# Información sobre cómo tomar la foto correctamente
with st.expander("¿Cómo tomar la FOTO correctamente?"):
    st.subheader("Coloca el producto correctamente en la cámara")
    for video_file in ["./videos/SI.mp4", "./videos/NO.mp4"]:
        try:
            with open(video_file, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
        except FileNotFoundError:
            st.error(f"El archivo de video no se encontró en la ruta: {video_file}")

