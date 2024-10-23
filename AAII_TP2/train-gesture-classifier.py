import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Función para añadir ruido gaussiano a los puntos clave
def add_noise(landmarks, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, size=landmarks.shape)
    return landmarks + noise

# Función para rotar los puntos clave aleatoriamente
def rotate(landmarks, angle_range=20):
    angle = np.radians(np.random.uniform(-angle_range, angle_range))
    center = np.mean(landmarks, axis=0)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    landmarks_rotated = np.dot(landmarks - center, rotation_matrix) + center
    return landmarks_rotated

# Función para trasladar los puntos clave
def translate(landmarks, max_translation=0.1):
    translation = np.random.uniform(-max_translation, max_translation, size=(2,))
    return landmarks + translation

# Función para escalar los puntos clave
def scale(landmarks, scale_range=0.2):
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    center = np.mean(landmarks, axis=0)
    return (landmarks - center) * scale_factor + center

# Función para aplicar una transformación de data augmentation aleatoria a los puntos clave
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
X = np.load('mi_dataset_X.npy')
Y = np.load('mi_dataset_Y.npy')

# Verificar las dimensiones de los datos cargados
print("Datos cargados:")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Aplicar data augmentation solo en el conjunto de entrenamiento
X_train_augmented = np.array([augment_landmarks(sample) for sample in X_train])

# Combinar el dataset original de entrenamiento con el dataset aumentado
X_train_combined = np.concatenate([X_train, X_train_augmented], axis=0)
Y_train_combined = np.concatenate([Y_train, Y_train], axis=0)

# Aplanar los puntos clave para la red neuronal
X_train_flattened = X_train_combined.reshape(X_train_combined.shape[0], 42)
X_test_flattened = X_test.reshape(X_test.shape[0], 42)

# Convertir las etiquetas a formato categórico (one-hot encoding)
Y_train_categorical = to_categorical(Y_train_combined, num_classes=3)
Y_test_categorical = to_categorical(Y_test, num_classes=3)

# Definir el modelo de la red neuronal
model = models.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con el dataset de entrenamiento aumentado
print("Entrenando el modelo con dataset aumentado y Dropout...")
history = model.fit(X_train_flattened, Y_train_categorical, 
                    epochs=1500, batch_size=128, 
                    validation_data=(X_test_flattened, Y_test_categorical))

# Guardar el modelo entrenado
model.save('mi_modelo.h5')
print("Modelo guardado en 'mi_modelo.h5'.")

# Extraer los datos de la historia de entrenamiento y validación
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Definir el rango de épocas para graficar
epochs_range = range(1500)

# Graficar la precisión y la pérdida
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
