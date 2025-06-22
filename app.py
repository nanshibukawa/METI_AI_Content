import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # Necessary for loading custom layers
import numpy as np
import matplotlib.pyplot as plt
import os # To handle file paths

# --- Configuration (matching previous steps for model parameters) ---
latent_dim = 20
image_shape = (28, 28, 1)
# batch_size # Not directly used in app logic, but good to remember for training context

# --- Custom Sampling Layer (MUST be defined for loading the model) ---
# This class needs to be available in the global scope when Keras loads the model
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Model Loading (Cached for Streamlit efficiency) ---
@st.cache_resource # Use st.cache_resource for loading ML models and complex objects
def load_trained_decoder():
    save_dir = "./trained_vae_models" # Path to your saved models relative to app.py
    decoder_load_path = os.path.join(save_dir, "decoder_mnist_vae.keras")

    if not os.path.exists(decoder_load_path):
        st.error(f"Model file not found at: {decoder_load_path}")
        st.stop() # Stop the app if model is not found

    try:
        # Load the decoder model, specifying the custom object
        decoder = keras.models.load_model(decoder_load_path, custom_objects={'Sampling': Sampling})
        st.success("Trained Decoder model loaded successfully!")
        return decoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
# Load the decoder model once at the start of the app
decoder_model = load_trained_decoder()

# --- Image Generation Logic (from previous step, adapted) ---
# For a Streamlit app, we'd typically pre-compute/save these prototypes,
# or compute them dynamically if the app starts fast enough.
# For simplicity, and knowing it computed quickly, we can recalculate or load them.

# Re-define build_encoder here ONLY IF you need to re-compute prototypes on app startup
# This is less ideal, usually you'd save/load the 'average_latent_vectors'
# However, if 'encoder_mnist_vae.keras' is present and small enough, it's fine.
# For a faster app, you would pre-calculate average_latent_vectors and save them as a .npy file,
# then load that .npy file in the Streamlit app instead of the encoder.

# For this example, let's re-use the encoder to calculate prototypes, assuming it's available.
# In a production app, you'd likely save `average_latent_vectors` from your Colab training.
@st.cache_data # Cache the data loading and processing
def get_digit_prototypes(encoder_model_path):
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)

    # Need to load encoder if not already available in the app context
    # If using @st.cache_resource for encoder above, pass it directly
    try:
        loaded_encoder = keras.models.load_model(encoder_model_path, custom_objects={'Sampling': Sampling})
    except Exception as e:
        st.error(f"Error loading encoder for prototypes: {e}")
        return {}

    digit_latent_means = {i: [] for i in range(10)}
    num_samples_for_prototypes = 10000 # Use same number as during training

    for i in range(num_samples_for_prototypes):
        img = x_train[i:i+1]
        label = y_train[i]
        z_mean, _, _ = loaded_encoder(img)
        digit_latent_means[label].append(z_mean.numpy().squeeze())

    average_latent_vectors = {}
    for digit, latent_vectors in digit_latent_means.items():
        if latent_vectors:
            average_latent_vectors[digit] = np.mean(latent_vectors, axis=0)
    return average_latent_vectors

# Get prototypes (encoder_load_path points to where encoder.keras should be in your repo)
average_latent_vectors = get_digit_prototypes(encoder_load_path)


def generate_digit_images(decoder, digit, num_images=5, latent_noise_scale=0.5):
    """
    Generates images of a specific digit using the decoder and latent prototypes.
    """
    if digit not in average_latent_vectors:
        st.warning(f"Prototype for digit {digit} not found. Cannot generate.")
        return []

    prototype_latent_vector = average_latent_vectors[digit]
    generated_images = []

    for _ in range(num_images):
        noise = np.random.normal(loc=0.0, scale=latent_noise_scale, size=latent_dim)
        noisy_latent_vector = prototype_latent_vector + noise
        
        noisy_latent_vector_tensor = tf.convert_to_tensor(noisy_latent_vector, dtype=tf.float32)
        noisy_latent_vector_tensor = tf.expand_dims(noisy_latent_vector_tensor, axis=0) # Add batch dim

        generated_img = decoder(noisy_latent_vector_tensor)
        generated_images.append(generated_img.numpy().squeeze()) # To numpy, remove batch/channel dims

    return generated_images

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Handwritten Digit Generator")

st.title("✍️ Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained VAE model.")

# User input for digit selection
selected_digit = st.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10)),
    index=7 # Default to 7 as per example
)

# Slider for controlling noise (optional, but good for exploration)
noise_scale = st.slider(
    "Adjust Generation Diversity (Noise Scale):",
    min_value=0.0, max_value=2.0, value=0.5, step=0.1,
    help="Higher value adds more randomness, lower value makes images more similar."
)

if st.button("Generate Images"):
    if decoder_model is None:
        st.error("Model not loaded. Please check logs.")
    else:
        st.subheader(f"Generated images of digit {selected_digit}")
        
        # Use st.spinner for a loading message
        with st.spinner(f'Generating 5 images for digit {selected_digit}...'):
            generated_imgs = generate_digit_images(decoder_model, selected_digit, num_images=5, latent_noise_scale=noise_scale)
            
            if generated_imgs:
                # Display images in columns
                cols = st.columns(5)
                for i, img in enumerate(generated_imgs):
                    with cols[i]:
                        st.image(img, caption=f"Sample {i+1}", use_column_width=True, clamp=True)
            else:
                st.warning("Could not generate images. Check model loading or prototype generation.")

st.markdown("---")
st.markdown("This app uses a Variational Autoencoder (VAE) trained from scratch on the MNIST dataset.")