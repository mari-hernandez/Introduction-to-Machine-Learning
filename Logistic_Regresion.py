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

# Crear un modelo de regresión logística usando Keras
model = Sequential([
    # Aplanar las imágenes de 28x28 píxeles en un vector de 784 elementos
    Flatten(input_shape=(28, 28)),
    # Agregar una capa completamente conectada (Dense) con 10 unidades y activación softmax
    # La activación softmax se utiliza para asignar probabilidades a las 10 clases de dígitos (0-9)
    Dense(10, activation='softmax')  # 10 unidades en la capa de salida para 10 clases en MNIST
])

# Compilar el modelo
model.compile(optimizer='adam',
              # Usamos la función de pérdida 'sparse_categorical_crossentropy' ya que tenemos etiquetas enteras
              loss='sparse_categorical_crossentropy',
              # Métrica para evaluar el rendimiento del modelo
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluar la precisión en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
# Imprimir la precisión en el conjunto de prueba
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')
