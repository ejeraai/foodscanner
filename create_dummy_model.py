import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====================================
# BUAT MODEL DUMMY UNTUK TESTING
# ====================================
# Model ini TIDAK AKURAT, hanya untuk testing aplikasi!

print("🔧 Membuat dummy model untuk testing...")

# Build simple model
model = keras.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save model (tanpa training)
model.save('food_classifier_model.h5')

print("✅ Dummy model berhasil dibuat!")
print("⚠️  WARNING: Model ini BELUM dilatih!")
print("⚠️  Prediksi akan random/tidak akurat")
print("⚠️  Hanya untuk testing aplikasi web saja")
print("\n💡 Untuk hasil akurat, gunakan model yang sudah dilatih dengan dataset yang proper")