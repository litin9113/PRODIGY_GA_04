import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load and preprocess the images
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    w = tf.shape(image)[1]
    input_image = image[:, w // 2:, :]
    real_image = image[:, :w // 2, :]
    input_image = tf.cast(input_image, tf.float32) / 255.0
    real_image = tf.cast(real_image, tf.float32) / 255.0
    return input_image, real_image

# Load dataset
def load_dataset(path):
    dataset = tf.data.Dataset.list_files(str(path))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

# More complex U-Net architecture
def complex_unet():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    # Encoder
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Bottleneck
    x = layers.Conv2D(512, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # Decoder
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Loss function
loss_object = tf.keras.losses.MeanSquaredError()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Training function
def train(generator, dataset, epochs=20):
    for epoch in range(epochs):
        for input_image, real_image in dataset:
            with tf.GradientTape() as tape:
                generated_image = generator(input_image, training=True)
                loss = loss_object(real_image, generated_image)  # Calculate loss

            gradients = tape.gradient(loss, generator.trainable_variables)
            optimizer.apply_gradients(zip(gradients, generator.trainable_variables))  # Update weights

            print(f"Epoch: {epoch+1}, Loss: {loss.numpy():.4f}")  # Print loss for monitoring

# Load dataset
PATH = r"C:\Users\litin\.keras\datasets\facades_extracted\facades\train\*.jpg"  # Update with your dataset path
train_dataset = load_dataset(PATH)

# Create and compile the model
generator = complex_unet()

# Train the model (increase epochs for better results)
train(generator, train_dataset, epochs=50)

# Test with a sample image
for input_image, real_image in train_dataset.take(1):
    generated_image = generator(input_image)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_image[0])
    plt.subplot(1, 3, 2)
    plt.title("Generated Image")
    plt.imshow(generated_image[0])
    plt.subplot(1, 3, 3)
    plt.title("Real Image")
    plt.imshow(real_image[0])
    plt.show()
