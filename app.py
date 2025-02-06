import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image 
import numpy as np

# Cargar el modelo
model = load_model('/fashion_mnist.keras')

# Crear interfaz de usuario
st.title('Clasificador de Fashion MNIST')
st.write("Sube una imagen en escala de grises de 28x28 píxeles")

upload_file = st.file_uploader("Sube una imagen en escala de grises de 28x28 píxeles", type=["png", "jpg", "jpeg"])
if upload_file is not None:
    image = Image.open(upload_file).convert('L') # Convertir RGB a blanco y negro
    image = image.resize((28, 28)) # Redimensionar la imagen a 28x28 píxeles
    img_array = np.array(image) / 255.0 # Normalizar la imagen
    img_array = img_array.reshape(1, 28, 28, 1) # El primer 1 indica que solo hay una imagen, luego las dimensiones
    # el último 1 indica que sólo hay un canal de color
    # Muestra la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)
    # Hacer la predicción
    prediction = model.predict(img_array)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Mostrar la predicción
    st.write("Predicción:", classes[np.argmax(prediction)])
