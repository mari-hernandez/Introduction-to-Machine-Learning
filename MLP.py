import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada a valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Crear un modelo de MLP con dos capas ocultas
model = Sequential([
    Flatten(input_shape=(28, 28)),
    # Capa oculta 1: 500 unidades con activación ReLU
    Dense(500, activation='relu'),
    # Capa oculta 2: 100 unidades con activación ReLU
    Dense(100, activation='relu'),
    # Capa de salida: 10 unidades con activación softmax
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluar la precisión en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
# Imprimir la precisión en el conjunto de prueba
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')
