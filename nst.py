import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and Preprocess Images
# -------------------------------
def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Paths to your images
content_path = "C:/Users/BhargaviMandala/Downloads/flower.jpg"
style_path = "C:/Users/BhargaviMandala/Downloads/flowerart.jpg"

content_image = load_image(content_path)
style_image = load_image(style_path)

# -------------------------------
# 2. Load TensorFlow Hub Model
# -------------------------------
print("Loading Style Transfer Model...")
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# -------------------------------
# 3. Apply Style Transfer
# -------------------------------
stylized_image = model(content_image, style_image)[0]  # [0] to get the output image

# -------------------------------
# 4. Convert and Save Output
# -------------------------------
output_image = tf.squeeze(stylized_image).numpy()  # Remove batch dimension
output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8

# Save output
output_pil = Image.fromarray(output_image)
output_path = "C:/Users/BhargaviMandala/Downloads/stylized_output.jpg"
output_pil.save(output_path)
print(f"Stylized image saved to: {output_path}")

# -------------------------------
# 5. Display the Image
# -------------------------------
plt.imshow(output_pil)
plt.title("Stylized Image")
plt.axis("off")
plt.show()

