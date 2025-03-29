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

# Configura la página de Streamlit con título, icono y estado inicial de la barra lateral
st.set_page_config(
    page_title="Reconocimiento de Objetos",  # Título de la página
    page_icon=":smile:",  # Icono de la página
    initial_sidebar_state='auto'  # Estado inicial de la barra lateral
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

# Define una función para cargar el modelo de inteligencia artificial con caché para optimizar el rendimiento
@st.cache_resource
def load_model():
    model_path = "./modelo_entrenado.h5"  # Ruta del modelo entrenado
    
    # Verifica si el archivo del modelo existe en la ruta especificada
    if not os.path.exists(model_path):
        st.error("Error: No se encontró el modelo entrenado. Verifica la ruta.")
        return None
    try:
        # Carga el modelo sin compilar para evitar posibles errores
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        # Muestra un mensaje de error en caso de fallo al cargar el modelo
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Muestra un mensaje de carga mientras se ejecuta la función de carga del modelo
with st.spinner('Cargando modelo...'):
    model = load_model()

# Cargar nombres de clases desde un archivo externo
class_names = []
try:
    with open("claseIA.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip().lower() for line in f.readlines()]  # Lee y almacena los nombres de las clases en minúsculas
    if not class_names:
        st.error("El archivo claseIA.txt está vacío.")  # Muestra un error si el archivo está vacío
except FileNotFoundError:
    st.error("No se encontró el archivo claseIA.txt.")  # Muestra un error si el archivo no se encuentra

# Cargar descripciones de objetos desde un archivo externo
descripcion_dict = {}
try:
    with open("proma.txt", "r", encoding="utf-8") as f:
        for line in f:
            partes = line.strip().split(":", 1)  # Divide cada línea en clave y descripción
            if len(partes) == 2:
                clave = partes[0].strip().lower()  # Convierte la clave a minúsculas
                descripcion = partes[1].strip()  # Extrae la descripción
                descripcion_dict[clave] = descripcion  # Almacena la clave y descripción en el diccionario
except FileNotFoundError:
    st.error("No se encontró el archivo proma.txt.")  # Muestra un error si el archivo no se encuentra

# Configuración de la barra lateral en la aplicación web
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")  # Muestra un video en la barra lateral
    st.title("Reconocimiento de imagen")  # Título en la barra lateral
    st.subheader("Identificación de objetos con VGG16")  # Subtítulo en la barra lateral
    
    # Slider para seleccionar el nivel de confianza del modelo (0-100%)
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100  # Se normaliza entre 0 y 1

# Muestra una imagen en la interfaz principal de la aplicación
st.image('smartregionlab2.jpeg')  # Imagen representativa del proyecto
st.title("Modelo de Identificación de Objetos - Smart Regions Center")  # Título principal
st.write("Desarrollo del Proyecto de Ciencia de Datos con Redes Convolucionales")  # Descripción del proyecto
def preprocess_image(image):
    # Verifica si la imagen no está en modo RGB y la convierte si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensiona la imagen a 224x224 píxeles, compatible con el modelo
    image = image.resize((224, 224))
    
    # Convierte la imagen en un array de NumPy
    image_array = np.array(image)
    
    # Expande las dimensiones para que sea compatible con el modelo de predicción
    image_array = np.expand_dims(image_array, axis=0)
    
    # Aplica el preprocesamiento requerido por el modelo
    image_array = preprocess_input(image_array)
    
    return image_array

def import_and_predict(image, model, class_names):
    # Verifica si el modelo está cargado
    if model is None:
        return "Modelo no cargado", 0.0
    
    # Preprocesa la imagen antes de realizar la predicción
    image = preprocess_image(image)
    
    # Realiza la predicción con el modelo
    prediction = model.predict(image)
    
    # Obtiene el índice de la clase con mayor probabilidad
    index = np.argmax(prediction[0])
    
    # Obtiene la confianza de la predicción
    confidence = np.max(prediction[0])
    
    # Verifica si el índice está dentro de los nombres de clase disponibles
    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Desconocido"
    
    return class_name, confidence

def generar_audio(texto):
    """Genera audio asegurando que siempre haya contenido."""
    # Si el texto está vacío, proporciona un mensaje predeterminado
    if not texto.strip():
        texto = "No se encontró información para este objeto."
    
    # Convierte el texto en audio usando gTTS
    tts = gTTS(text=texto, lang='es')
    
    # Crea un buffer en memoria para almacenar el archivo de audio
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    
    # Reinicia la posición del buffer para su lectura
    mp3_fp.seek(0)
    
    return mp3_fp

def reproducir_audio(mp3_fp):
    """Reproduce el audio generado en Streamlit."""
    # Lee los bytes del archivo de audio
    audio_bytes = mp3_fp.read()
    
    # Convierte el audio a formato base64 para incrustarlo en HTML
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    # Genera una etiqueta HTML para la reproducción automática del audio
    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    
    # Inserta el HTML en la aplicación Streamlit
    st.markdown(audio_html, unsafe_allow_html=True)

# Captura una imagen desde la cámara o permite la carga de un archivo
img_file_buffer = st.camera_input("Capture una foto para identificar el objeto") or \
                  st.file_uploader("Cargar imagen desde archivo", type=["jpg", "jpeg", "png"])

resultado = "No se ha procesado ninguna imagen."

# Si no hay imagen cargada, permite ingresar una URL
if img_file_buffer is None:
    image_url = st.text_input("O ingrese la URL de la imagen")
    if image_url:
        try:
            # Descarga la imagen desde la URL proporcionada
            response = requests.get(image_url)
            img_file_buffer = BytesIO(response.content)
        except Exception as e:
            st.error(f"Error al cargar la imagen desde la URL: {e}")

# Si hay una imagen cargada y el modelo está disponible
if img_file_buffer and model:
    try:
        # Abre la imagen con PIL
        image = Image.open(img_file_buffer)
        
        # Muestra la imagen en la interfaz de Streamlit
        st.image(image, use_column_width=True)
        
        # Realiza la predicción con el modelo
        class_name, confidence_score = import_and_predict(image, model, class_names)
        
        # Obtiene la descripción de la clase detectada
        descripcion = descripcion_dict.get(class_name, "No hay información disponible para este objeto.")
        
        # Verifica si la confianza de la predicción supera el umbral establecido
        if confidence_score > confianza:
            resultado = f"Objeto Detectado: {class_name.capitalize()}\n"
            resultado += f"Confianza: {100 * confidence_score:.2f}%\n\n"
            resultado += f"Descripción: {descripcion}"
            
            # Elimina caracteres no deseados del resultado
            resultado_limpio = resultado.replace('*', '').replace('_', '').replace('/', '')
            
            # Muestra los resultados en la interfaz de Streamlit
            st.subheader(f"Tipo de Objeto: {class_name.capitalize()}")
            st.text(f"Confianza: {100 * confidence_score:.2f}%")
            st.write(f"Descripción: {descripcion}")
        else:
            resultado = "No se pudo determinar el tipo de objeto"
            st.text(resultado)
        
        # Genera y reproduce el audio con la descripción del objeto
        mp3_fp1 = generar_audio(resultado_limpio)
        reproducir_audio(mp3_fp1)
        mp3_fp = generar_audio(descripcion)
        reproducir_audio(mp3_fp1)
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
else:
    st.text("Por favor, cargue una imagen usando una de las opciones anteriores.")
