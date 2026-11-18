
import os, time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("cikti", exist_ok=True)
tf.keras.utils.set_random_seed(42)

# BÖLÜM 3 - Konvolüsyon katmanı

print("\n===== BÖLÜM 3: KONVOLÜSYON =====")
conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                              activation='relu', kernel_initializer='he_normal')
dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("İlk filtre:\n", np.round(weights[:, :, 0, 0], 4))


fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[:, :, 0, i], cmap='gray')
    ax.set_title(f"F{i+1}", fontsize=9)
    ax.axis('off')
plt.suptitle("16 filtrenin başlangıç ağırlıkları (3x3, he_normal)")
plt.tight_layout()
plt.savefig("cikti/filtreler.png", dpi=150)
plt.close()


# BÖLÜM 4 - Max Pooling

print("\n===== BÖLÜM 4: MAX POOLING =====")
input_data = tf.constant([[[1., 3., 2., 4.],
                           [5., 6., 1., 2.],
                           [7., 2., 8., 3.],
                           [4., 1., 5., 9.]]])
input_data = tf.reshape(input_data, (1, 4, 4, 1))
pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
pooled = pool(input_data)
print("Giriş:\n", input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:\n", pooled.numpy()[0, :, :, 0])
print("Giriş boyutu:", input_data.shape, "-> Çıkış boyutu:", pooled.shape)

# BÖLÜM 5 - ReLU ve Softmax

print("\n===== BÖLÜM 5: ReLU ve SOFTMAX =====")
x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
y = tf.keras.layers.ReLU()(x)
print("Giriş:", x.numpy())
print("ReLU çıkışı:", y.numpy())

logits = tf.constant([[2.0, 1.0, 0.1]])
probabilities = tf.keras.layers.Softmax()(logits)
print("Softmax sonuçları:", np.round(probabilities.numpy(), 4))
print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())
print("En olası sınıf (argmax):", int(tf.argmax(probabilities, axis=1)[0]))

# BÖLÜM 6 - Fashion-MNIST ile CNN

print("\n===== BÖLÜM 6: CNN MODELİ =====")
try:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
except Exception:
    print("Veri indirilemedi, yerel dosya kullanılıyor.")
    d = np.load("data/fashion_mnist.npz")
    x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)


def build_model(first_filters=8):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])


def train(first_filters):
    tf.keras.utils.set_random_seed(42)          
    model = build_model(first_filters)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    t0 = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                        validation_split=0.1, verbose=2)
    sure = time.time() - t0
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return model, history, sure, test_loss, test_acc



model, history, sure1, test_loss, test_accuracy = train(8)
model.summary()
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
plt.savefig("cikti/egitim_grafikleri.png", dpi=150)
plt.close()


print("\nKatman çıkış boyutları:")
for layer in model.layers:
    print(f"  {layer.name:<20} {layer.output.shape}  parametre: {layer.count_params()}")


siniflar = ['T-shirt', 'Pantolon', 'Kazak', 'Elbise', 'Ceket',
            'Sandalet', 'Gömlek', 'Spor ayakkabı', 'Çanta', 'Bot']
pred = model.predict(x_test[:8], verbose=0)
fig, axes = plt.subplots(1, 8, figsize=(13, 2.4))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i, :, :, 0], cmap='gray'); ax.axis('off')
    ok = np.argmax(pred[i]) == y_test[i]
    ax.set_title(f"{siniflar[np.argmax(pred[i])]}\n%{100*pred[i].max():.0f}",
                 fontsize=8, color='green' if ok else 'red')
plt.tight_layout()
plt.savefig("cikti/ornek_tahminler.png", dpi=150)
plt.close()

# BÖLÜM 8 - Deney: 8 vs 16 filtre

print("\n BÖLÜM 8: DENEY (16 filtre) ")
_, history2, sure2, _, test_acc2 = train(16)

print("\n{:<20}{:>12}{:>12}".format("Ölçüt", "Deney 1", "Deney 2"))
print("{:<20}{:>12}{:>12}".format("Filtre sayısı", 8, 16))
print("{:<20}{:>12.4f}{:>12.4f}".format("Eğitim doğruluğu",
      history.history['accuracy'][-1], history2.history['accuracy'][-1]))
print("{:<20}{:>12.4f}{:>12.4f}".format("Test doğruluğu", test_accuracy, test_acc2))
print("{:<20}{:>11.1f}s{:>11.1f}s".format("Eğitim süresi", sure1, sure2))
