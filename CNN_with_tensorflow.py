import tensorflow as tf
import matplotlib.pyplot as plt

# Download and extract a sample dataset
image_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
path_to_zip = tf.keras.utils.get_file(
    "flower_photos.tgz", origin=image_url, extract=True
)

image_path = path_to_zip.replace(".tgz", "/daisy/3475870145_685a19116d.jpg")
input_image = tf.keras.utils.load_img(image_path, target_size=(224, 224))

# Preprocess the image
input_array = tf.keras.utils.img_to_array(input_image) / 255.0  # Normalize to range 0-1
input_tensor = tf.expand_dims(input_array, axis=0)  # Add a batch dimension

# Create a convolutional layer
conv_layer = tf.keras.layers.Conv2D(
    filters=8,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding="same",
    activation="relu",
)

# Generate activation maps
output_tensor = conv_layer(input_tensor)

# Display the input image
plt.figure(figsize=(6, 6))
plt.imshow(input_array)
plt.axis("off")
plt.show()

# Visualize the activation maps
activation_maps = output_tensor[0].numpy()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(activation_maps[:, :, i], cmap="viridis")
    ax.axis("off")
plt.show()
