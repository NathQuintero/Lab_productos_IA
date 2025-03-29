import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import warnings
from gtts import gTTS
import base64
import time

# Ignore warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="CamApp - Reconocimiento de Objetos",
    page_icon="🔍",
    initial_sidebar_state='collapsed',
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f5f7f9;
    }
    
    /* Card styling */
    .stCard {
        border-radius: 15px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.08);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        transition: transform 0.3s ease;
    }
    .stCard:hover {
        transform: translateY(-5px);
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        text-align: center;
        padding: 40px;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    
    /* Button styling */
    .icon-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #4CAF50;
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin: 10px;
        text-decoration: none;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }
    .icon-button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #f6d365, #fda085);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin-top: 20px;
    }
    
    /* Learn more button */
    .learn-more-btn {
        background-color: #2196F3;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        margin-top: 15px;
        font-weight: bold;
    }
    .learn-more-btn:hover {
        background-color: #0b7dda;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'show_description' not in st.session_state:
    st.session_state.show_description = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'class_name' not in st.session_state:
    st.session_state.class_name = ""
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = 0.0
if 'descripcion' not in st.session_state:
    st.session_state.descripcion = ""

# Audio functions
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

# Model loading function
@st.cache_resource
def load_model():
    model_path = "./modelo_entrenado.h5"
    if not os.path.exists(model_path):
        st.error("Error: No se encontró el modelo entrenado. Verifica la ruta.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Load model
model = load_model()

# Load class names
class_names = []
try:
    with open("claseIA.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip().lower() for line in f.readlines()]
    if not class_names:
        st.error("El archivo claseIA.txt está vacío.")
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt.")

# Load object descriptions
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

# Image processing functions
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

# Navigation functions
def go_to_main():
    st.session_state.page = 'main'
    texto = "Empecemos. Por favor, selecciona una opción para identificar un objeto."
    mp3_fp = generar_audio(texto)
    reproducir_audio(mp3_fp)

def toggle_description():
    st.session_state.show_description = not st.session_state.show_description
    if st.session_state.show_description:
        mp3_fp = generar_audio(st.session_state.descripcion)
        reproducir_audio(mp3_fp)

# Welcome page
if st.session_state.page == 'welcome':
    st.markdown("""
    <div class="welcome-card">
        <h1>¡Bienvenido a CamApp!</h1>
        <p style="font-size: 1.2rem; margin: 20px 0;">
            CamApp es una aplicación de inteligencia artificial que te permite identificar objetos 
            a través de imágenes. Simplemente toma una foto, sube un archivo o proporciona una URL 
            y nuestro modelo de IA te dirá qué objeto es con una descripción detallada.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <button class="icon-button" id="start-button" onclick="startApp()">
                <span style="margin-right: 10px;">▶️</span> EMPEZAR
            </button>
        </div>
        
        <script>
        function startApp() {
            window.parent.postMessage({type: "streamlit:setComponentValue", value: true}, "*");
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Hidden button to trigger the state change
        if st.button("Empezar", key="start_hidden", help="Iniciar la aplicación"):
            go_to_main()
    
    # Play welcome audio when page loads
    welcome_text = "¡Bienvenido a CamApp! Soy tu asistente de inteligencia artificial. Esta aplicación te permite identificar objetos a través de imágenes. Presiona el botón Empezar para comenzar."
    mp3_fp = generar_audio(welcome_text)
    reproducir_audio(mp3_fp)

# Main application page
elif st.session_state.page == 'main':
    # App header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>CamApp - Reconocimiento de Objetos</h1>
        <p>Selecciona una de las siguientes opciones para identificar un objeto</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option buttons with icons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stCard" style="text-align: center;">
            <h3>📷 Tomar Foto</h3>
            <p>Usa la cámara para capturar una imagen</p>
        </div>
        """, unsafe_allow_html=True)
        camera_input = st.camera_input("", label_visibility="collapsed")
        
        if camera_input:
            with st.spinner('Procesando imagen...'):
                image = Image.open(camera_input)
                st.session_state.class_name, st.session_state.confidence_score = import_and_predict(image, model, class_names)
                st.session_state.descripcion = descripcion_dict.get(st.session_state.class_name, "No hay información disponible para este objeto.")
                st.session_state.prediction_made = True
                
                # Generate audio for the result
                resultado = f"Objeto Detectado: {st.session_state.class_name.capitalize()}. Confianza: {100 * st.session_state.confidence_score:.2f}%"
                mp3_fp = generar_audio(resultado)
                reproducir_audio(mp3_fp)
    
    with col2:
        st.markdown("""
        <div class="stCard" style="text-align: center;">
            <h3>📁 Subir Archivo</h3>
            <p>Sube una imagen desde tu dispositivo</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            with st.spinner('Procesando imagen...'):
                image = Image.open(uploaded_file)
                st.session_state.class_name, st.session_state.confidence_score = import_and_predict(image, model, class_names)
                st.session_state.descripcion = descripcion_dict.get(st.session_state.class_name, "No hay información disponible para este objeto.")
                st.session_state.prediction_made = True
                
                # Generate audio for the result
                resultado = f"Objeto Detectado: {st.session_state.class_name.capitalize()}. Confianza: {100 * st.session_state.confidence_score:.2f}%"
                mp3_fp = generar_audio(resultado)
                reproducir_audio(mp3_fp)
    
    with col3:
        st.markdown("""
        <div class="stCard" style="text-align: center;">
            <h3>🔗 URL de Imagen</h3>
            <p>Proporciona un enlace a una imagen</p>
        </div>
        """, unsafe_allow_html=True)
        image_url = st.text_input("", placeholder="Ingresa la URL de la imagen", label_visibility="collapsed")
        
        if image_url:
            try:
                with st.spinner('Descargando y procesando imagen...'):
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))
                    st.session_state.class_name, st.session_state.confidence_score = import_and_predict(image, model, class_names)
                    st.session_state.descripcion = descripcion_dict.get(st.session_state.class_name, "No hay información disponible para este objeto.")
                    st.session_state.prediction_made = True
                    
                    # Generate audio for the result
                    resultado = f"Objeto Detectado: {st.session_state.class_name.capitalize()}. Confianza: {100 * st.session_state.confidence_score:.2f}%"
                    mp3_fp = generar_audio(resultado)
                    reproducir_audio(mp3_fp)
            except Exception as e:
                st.error(f"Error al cargar la imagen desde la URL: {e}")
    
    # Display prediction results
    if st.session_state.prediction_made:
        st.markdown("""
        <div class="result-card">
            <h2 style="text-align: center; margin-bottom: 20px;">Resultado del Análisis</h2>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if camera_input:
                st.image(camera_input, use_column_width=True)
            elif uploaded_file:
                st.image(uploaded_file, use_column_width=True)
            elif image_url:
                st.image(image_url, use_column_width=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; background-color: rgba(255, 255, 255, 0.2); border-radius: 10px;">
                <h3>Objeto Detectado: {st.session_state.class_name.capitalize()}</h3>
                <div style="background-color: rgba(0, 0, 0, 0.1); height: 30px; border-radius: 15px; margin: 15px 0; position: relative;">
                    <div style="background-color: #4CAF50; width: {100 * st.session_state.confidence_score:.1f}%; height: 100%; border-radius: 15px; position: absolute; top: 0; left: 0;"></div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-weight: bold;">
                        Confianza: {100 * st.session_state.confidence_score:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Learn more button
            st.markdown("""
                <button class="learn-more-btn" id="learn-more-btn" onclick="toggleDescription()">
                    Conocer más de este producto
                </button>
                
                <script>
                function toggleDescription() {
                    window.parent.postMessage({type: "streamlit:setComponentValue", value: true}, "*");
                }
                </script>
            """, unsafe_allow_html=True)
            
            # Hidden button to trigger the state change
            if st.button("Conocer más", key="learn_more_hidden", help="Ver descripción detallada"):
                toggle_description()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show description if button was clicked
        if st.session_state.show_description:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px; color: #333;">
                <h3>Descripción del Producto</h3>
                <p>{st.session_state.descripcion}</p>
            </div>
            """, unsafe_allow_html=True)

# Sidebar with additional information
with st.sidebar:
    st.title("Smart Regions Center")
    st.image('smartregionlab2.jpeg')
    st.subheader("Acerca de CamApp")
    st.write("CamApp utiliza un modelo de inteligencia artificial basado en VGG16 para identificar objetos en imágenes.")
    
    # Confidence slider
    st.subheader("Configuración")
    confianza = st.slider("Nivel de confianza mínimo", 0, 100, 50) / 100
    
    # Reset button
    if st.button("Volver al inicio"):
        st.session_state.page = 'welcome'
        st.session_state.prediction_made = False
        st.session_state.show_description = False
        st.experimental_rerun()
