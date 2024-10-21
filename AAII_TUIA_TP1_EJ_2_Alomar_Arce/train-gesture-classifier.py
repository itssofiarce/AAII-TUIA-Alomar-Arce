import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Función para añadir ruido gaussiano a los puntos clave.
# Esto genera pequeñas variaciones en las coordenadas para mejorar la robustez del modelo.
def add_noise(landmarks, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, size=landmarks.shape)
    return landmarks + noise

# Función para rotar los puntos clave aleatoriamente.
# El ángulo de rotación se elige dentro de un rango especificado (por defecto 20 grados).
def rotate(landmarks, angle_range=20):
    angle = np.radians(np.random.uniform(-angle_range, angle_range))
    center = np.mean(landmarks, axis=0)  # Centro de los puntos clave
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    # Aplicar la rotación a los puntos clave, usando el centro como pivote
    landmarks_rotated = np.dot(landmarks - center, rotation_matrix) + center
    return landmarks_rotated

# Función para trasladar (mover) los puntos clave.
# La traslación se realiza en x e y dentro de un rango máximo especificado.
def translate(landmarks, max_translation=0.1):
    translation = np.random.uniform(-max_translation, max_translation, size=(2,))
    return landmarks + translation

# Función para escalar los puntos clave.
# Se aplica un factor de escala aleatorio, que ajusta el tamaño relativo de los puntos clave.
def scale(landmarks, scale_range=0.2):
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    center = np.mean(landmarks, axis=0)  # Centro de los puntos clave
    # Aplicar la escala manteniendo el centro fijo
    return (landmarks - center) * scale_factor + center

# Función para aplicar una transformación de data augmentation aleatoria a los puntos clave.
# Escoge entre añadir ruido, rotar, trasladar o escalar.
def augment_landmarks(landmarks):
    augmentation_choice = np.random.choice(['noise', 'rotate', 'translate', 'scale'])
    
    if augmentation_choice == 'noise':
        landmarks = add_noise(landmarks)
    elif augmentation_choice == 'rotate':
        landmarks = rotate(landmarks)
    elif augmentation_choice == 'translate':
        landmarks = translate(landmarks)
    elif augmentation_choice == 'scale':
        landmarks = scale(landmarks)
    
    return landmarks

# Cargar los datos de puntos clave y etiquetas desde archivos .npy
# X contiene las coordenadas x e y de los puntos clave, Y contiene las etiquetas (piedra, papel, tijeras)
X = np.load('AAII-TUIA-Alomar-Arce\AAII_TUIA_TP1_EJ_2_Alomar_Arce\mi_dataset_X.npy')  # Puntos clave (coordenadas x e y)
Y = np.load('AAII-TUIA-Alomar-Arce\AAII_TUIA_TP1_EJ_2_Alomar_Arce\mi_dataset_Y.npy')  # Etiquetas (0: piedra, 1: papel, 2: tijeras)

# Verificar las dimensiones de los datos cargados para asegurar que se cargaron correctamente.
print("Datos cargados:")
print("X shape:", X.shape)  # Dimensiones de los puntos clave
print("Y shape:", Y.shape)  # Dimensiones de las etiquetas

# Aplicar data augmentation a cada muestra de puntos clave.
X_augmented = np.array([augment_landmarks(sample) for sample in X])

# Combinar el dataset original con el dataset aumentado.
# Esto duplica el tamaño del dataset con nuevas muestras aumentadas.
X_combined = np.concatenate([X, X_augmented], axis=0)
Y_combined = np.concatenate([Y, Y], axis=0)

# Aplanar los puntos clave para que tengan la forma adecuada para la red neuronal.
# En lugar de tener (21, 2), necesitamos (42,), es decir, 21 puntos clave x 2 coordenadas.
X_combined_flattened = X_combined.reshape(X_combined.shape[0], 42)

# Convertir las etiquetas a formato categórico (one-hot encoding) para el problema de clasificación.
Y_categorical = to_categorical(Y_combined, num_classes=3)

# Definir el modelo de la red neuronal con capas densas (fully connected) y Dropout para regularización.
model = models.Sequential([
    layers.Input(shape=(42,)),  # Entrada de 42 características (21 puntos clave x 2 coordenadas)
    layers.Dense(64, activation='relu'),  # Capa densa con 64 neuronas y activación ReLU
    layers.Dropout(0.5),  # Dropout del 50% para prevenir sobreajuste
    layers.Dense(64, activation='relu'),  # Otra capa densa con 64 neuronas
    layers.Dropout(0.4),  # Dropout del 40% para mayor regularización
    layers.Dense(3, activation='softmax')  # Capa de salida con 3 neuronas (una para cada clase: piedra, papel, tijeras)
])

# Compilar el modelo con el optimizador Adam y función de pérdida categórica para clasificación.
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # Función de pérdida para problemas de clasificación multiclase
              metrics=['accuracy'])  # Métrica de precisión

# Entrenar el modelo con el dataset aumentado.
# Se entrena por 500 épocas y se usa un 20% del conjunto de datos para validación.
print("Entrenando el modelo con dataset aumentado y Dropout...")
history = model.fit(X_combined_flattened, Y_categorical, epochs=1500, batch_size=128, validation_split=0.2)

# Guardar el modelo entrenado en un archivo .h5 para usarlo posteriormente.
model.save('mi_modelo.h5')
print("Modelo guardado en 'mi_modelo.h5'.")

# Extraer los datos de la historia de entrenamiento y validación (precisión y pérdida).
acc = history.history['accuracy']  # Precisión en el conjunto de entrenamiento
val_acc = history.history['val_accuracy']  # Precisión en el conjunto de validación
loss = history.history['loss']  # Pérdida en el conjunto de entrenamiento
val_loss = history.history['val_loss']  # Pérdida en el conjunto de validación

# Definir el rango de épocas para graficar
epochs_range = range(1500)

# Graficar la precisión y la pérdida tanto para el entrenamiento como para la validación.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')

# Guardar el gráfico en un archivo PNG
plt.savefig('graf_metricas_entrenamiento.png')

# Mostrar los gráficos
plt.show()
