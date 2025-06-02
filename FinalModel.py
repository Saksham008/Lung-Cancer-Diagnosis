#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

# ✅ Set page config FIRST
st.set_page_config(page_title="Lung Cancer Classifier", layout="centered")

# ---- Define and Register CBAM ----
@tf.keras.utils.register_keras_serializable()
class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = layers.Dense(channel // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)
        self.conv_spatial = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        # Channel Attention
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))

        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(inputs)
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        channel_attention = layers.Add()([avg_out, max_out])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        channel_refined = layers.Multiply()([inputs, channel_attention])

        # Spatial Attention
        avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(channel_refined)
        max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(channel_refined)
        concat = layers.Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])
        spatial_attention = self.conv_spatial(concat)
        refined_feature = layers.Multiply()([channel_refined, spatial_attention])
        return refined_feature

# ---- Load Model with Custom CBAM Layer ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fusion_xception_vgg16_cbam_best_model.keras",
        custom_objects={"CBAM": CBAM},
        compile=False
    )

model = load_model()
class_names = ['aca', 'n', 'scc']

# Define class information with clinical detail
class_info = {
    'aca': {
        'full_name': 'Adenocarcinoma (ACA)',
        'description': 'Adenocarcinoma is a type of lung cancer that originates in the glandular cells of the lungs. It is often diagnosed in non-smokers and can grow in the outer parts of the lung. Early detection is crucial to prevent metastasis. Consult a pulmonologist or oncologist immediately for proper diagnosis and treatment planning.'
    },
    'n': {
        'full_name': 'Normal (N)',
        'description': 'This indicates healthy lung tissue with no visible signs of malignancy. It’s always recommended to maintain regular checkups if there are symptoms or family history of lung disease.'
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma (SCC)',
        'description': 'Squamous Cell Carcinoma is a common type of non-small cell lung cancer. It usually arises in the central part of the lung or major airways. Smoking is a major risk factor. Patients should seek immediate medical evaluation and diagnostic imaging.'
    }
}

# ---- Streamlit UI ----
st.title("🩺 Lung Cancer Classifier")
st.write("Upload a CT scan image, and this AI system will classify the image as Normal, Adenocarcinoma, or Squamous Cell Carcinoma.")

uploaded_file = st.file_uploader("Upload a lung CT scan image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    input_dict = {
        'x_input': img_array,  # Xception input
        'v_input': img_array   # VGG16 input
    }

    prediction = model.predict(input_dict)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    full_name = class_info[predicted_class]['full_name']
    description = class_info[predicted_class]['description']

    st.markdown("### 🧠 **Predicted Class:** `" + predicted_class + "`")
    st.markdown("### 🔍 **Confidence:** `" + f"{confidence:.2f}%" + "`")
    st.markdown(f"**Full Name:** `{full_name}`")
    st.markdown(f"**Description:** {description}")

    st.subheader("🔢 Class-wise Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name.upper()}: {prediction[0][i]*100:.2f}%")

    # Sample Recommendations (Rule-based)
    if predicted_class == 'aca':
        st.info("📋 Recommendation: Refer to an oncologist for further imaging and biopsy confirmation.")
    elif predicted_class == 'scc':
        st.info("📋 Recommendation: Recommend bronchoscopy or PET scan for further evaluation.")
    elif predicted_class == 'n':
        st.success("✅ No signs of cancer detected. If symptoms persist, consider routine follow-up.")

    # Diagnostic Confidence Indicator
    if confidence < 85:
        st.warning("⚠️ Confidence is below 85%. Consider retesting or reviewing multiple CT slices.")

    st.info("📢 *This is a deep learning-based classification. Final diagnosis must be made by a certified medical professional.*")
    st.warning("🩺 *If you experience symptoms or have concerns, please consult a healthcare provider promptly.*")


# In[ ]:




