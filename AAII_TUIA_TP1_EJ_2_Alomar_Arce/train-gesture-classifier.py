import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Funciones de Data Augmentation
def add_noise(landmarks, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, size=landmarks.shape)
    return landmarks + noise

def rotate(landmarks, angle_range=20):
    angle = np.radians(np.random.uniform(-angle_range, angle_range))
    center = np.mean(landmarks, axis=0)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    landmarks_rotated = np.dot(landmarks - center, rotation_matrix) + center
    return landmarks_rotated

def translate(landmarks, max_translation=0.1):
    translation = np.random.uniform(-max_translation, max_translation, size=(2,))
    return landmarks + translation

def scale(landmarks, scale_range=0.2):
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    center = np.mean(landmarks, axis=0)
    return (landmarks - center) * scale_factor + center

# Función para aplicar Data Augmentation aleatoria
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

# Cargar los datos desde los archivos .npy (coordenadas x e y de los puntos clave y las etiquetas)
X = np.load('mi_dataset_X.npy')  # Puntos clave (coordenadas x e y)
Y = np.load('mi_dataset_Y.npy')  # Etiquetas (0: piedra, 1: papel, 2: tijeras)

# Verificar las dimensiones de los datos
print("Datos cargados:")
print("X shape:", X.shape)  # (64, 21, 2)
print("Y shape:", Y.shape)  # (64,)

# Aplicar Data Augmentation aleatorio a cada muestra
X_augmented = np.array([augment_landmarks(sample) for sample in X])

# Combinar el dataset original con el dataset aumentado
X_combined = np.concatenate([X, X_augmented], axis=0)
Y_combined = np.concatenate([Y, Y], axis=0)

# Aplanar los datos X_combined para que tengan la forma correcta (None, 42) en lugar de (None, 21, 2)
X_combined_flattened = X_combined.reshape(X_combined.shape[0], 42)

# Convertir las etiquetas Y a formato categórico (one-hot encoding) para clasificación
Y_categorical = to_categorical(Y_combined, num_classes=3)

# Crear el modelo de la red neuronal densa con Dropout
model = models.Sequential([
    layers.Input(shape=(42,)),  # 42 entradas (21 puntos x 2 coordenadas: x, y)
    layers.Dense(64, activation='relu'),  # Capa densa con 64 neuronas y activación ReLU
    layers.Dropout(0.5),  # Dropout para regularización
    layers.Dense(64, activation='relu'),  # Otra capa densa con 64 neuronas
    layers.Dropout(0.4),  # Otro Dropout
    layers.Dense(3, activation='softmax')  # Capa de salida con 3 neuronas (piedra, papel, tijeras) y activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # Función de pérdida para clasificación
              metrics=['accuracy'])

# Entrenar el modelo y almacenar el historial
print("Entrenando el modelo con dataset aumentado y Dropout...")
history = model.fit(X_combined_flattened, Y_categorical, epochs=400, batch_size=128, validation_split=0.2)

# Guardar el modelo entrenado en un archivo .h5
model.save('mi_modelo.h5')
print("Modelo guardado en 'mi_modelo.h5'.")

# Graficar las métricas de entrenamiento y validación
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(400)  # Número de épocas 

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

plt.show()
