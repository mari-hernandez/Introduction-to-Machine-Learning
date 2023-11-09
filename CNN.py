import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada a valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Agregar una dimensión para los canales de las imágenes (en escala de grises)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Crear un modelo de CNN de dos capas
model = Sequential([
    # Capa de convolución 1: 32 filtros de 5x5 con activación ReLU
    Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    # Capa de max pooling 1: reducción de 2x2
    MaxPooling2D((2, 2)),
    # Capa de convolución 2: 64 filtros de 5x5 con activación ReLU
    Conv2D(64, (5, 5), activation='relu'),
    # Capa de max pooling 2: reducción de 2x2
    MaxPooling2D((2, 2)),
    # Aplanar los datos para la capa totalmente conectada
    Flatten(),
    # Capa totalmente conectada 1: 256 unidades con activación ReLU
    Dense(256, activation='relu'),
    # Capa totalmente conectada 2: 10 unidades con activación softmax
    Dense(10, activation='softmax')
])

# Compilar el modelo con el optimizador Adam
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluar la precisión en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')

# 1. Comparación con regresión logística y MLP:
# - La precisión del CNN suele ser más alta que la regresión logística y el MLP en tareas de visión por computadora.
# - El tiempo de entrenamiento del CNN es generalmente mayor debido a su complejidad.

# 2. Número de parámetros entrenables en el CNN:
# - El número de parámetros entrenables en el CNN construido es de aproximadamente 316,066 parámetros.
# Capa de convolución 1: 32 filtros de 5x5 + 32 sesgos = 832 parámetros
# Capa de convolución 2: 64 filtros de 5x5 + 64 sesgos = 51,264 parámetros
# Capa totalmente conectada 1: 3136 entradas x 256 unidades + 256 sesgos = 803,072 parámetros
# Capa totalmente conectada 2: 256 entradas x 10 unidades + 10 sesgos = 2570 parámetros
# Total: 832 + 51,264 + 803,072 + 2570 = 316,066 parámetros

# 3. Cuándo usar una CNN versus una regresión logística o un MLP:
# - Debes considerar el tipo de problema que estás abordando y el tipo de datos que tienes:
#   - Usa una regresión logística cuando tengas un problema de clasificación binaria o una tarea de regresión simple.
#   - Utiliza un MLP (Multilayer Perceptron) cuando estés tratando con datos estructurados o problemas de clasificación/regresión
#     más complejos en los que no haya una estructura espacial en los datos.
#   - Opta por una CNN cuando estés trabajando con datos de imágenes o datos con una estructura espacial, como series de tiempo.
#     Las CNN están diseñadas para capturar patrones locales y jerárquicos en los datos, lo que las hace ideales para tareas
#     de visión por computadora, detección de objetos, segmentación de imágenes y más. También son eficaces en problemas de
#     procesamiento de señales y otros dominios con estructura espacial.
