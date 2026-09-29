import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Brain Tumor MRI Detection")

st.write(
    "Upload a brain MRI scan and the trained VGG16 model "
    "will predict the tumor class."
)

st.warning(
    "⚠️ This application is for educational/research purposes only "
    "and should not be used as a medical diagnosis."
)

# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = "brain_tumor_vgg16_fp16_gzip.h5"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


try:
    model = load_model()
    st.success("✅ VGG16 model loaded successfully")

except Exception as e:
    st.error("❌ Unable to load the model.")
    st.error(str(e))
    st.stop()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

# IMPORTANT:
# These MUST match the class order used during model training.

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# --------------------------------------------------
# IMAGE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Brain MRI Scan",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # --------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------

    st.subheader("Uploaded MRI Scan")

    # IMPORTANT:
    # Do NOT use width="100%"
    # Use width="stretch" with newer Streamlit versions.

    st.image(
        image,
        caption="Brain MRI Scan",
        width="stretch"
    )

    # --------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------

    img = image.resize((224, 224))

    img_array = np.array(img)

    # Convert to float32
    img_array = img_array.astype("float32")

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------

    if st.button("🔍 Analyze MRI", type="primary"):

        with st.spinner("Analyzing MRI scan..."):

            prediction = model.predict(
                img_array,
                verbose=0
            )

        # --------------------------------------------------
        # CHECK MODEL OUTPUT
        # --------------------------------------------------

        if prediction.ndim == 2 and prediction.shape[-1] == len(CLASS_NAMES):

            probabilities = prediction[0]

            predicted_index = int(np.argmax(probabilities))

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = float(
                probabilities[predicted_index] * 100
            )

        else:

            st.error(
                "❌ The model output does not match the "
                "4-class classification setup."
            )

            st.write(
                "Model output shape:",
                prediction.shape
            )

            st.stop()

        # --------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------

        st.subheader("Prediction Result")

        if predicted_class == "No Tumor":

            st.success(
                f"✅ Prediction: {predicted_class}"
            )

        else:

            st.error(
                f"⚠️ Prediction: {predicted_class}"
            )

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # --------------------------------------------------
        # CLASS PROBABILITIES
        # --------------------------------------------------

        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            probability_percent = float(
                probability * 100
            )

            st.write(
                f"**{class_name}:** "
                f"{probability_percent:.2f}%"
            )

            st.progress(
                min(max(float(probability), 0.0), 1.0)
            )

        # --------------------------------------------------
        # MEDICAL DISCLAIMER
        # --------------------------------------------------

        st.info(
            "ℹ️ This prediction is generated by a machine-learning "
            "model for educational/research purposes. It is not a "
            "medical diagnosis. Please consult a qualified medical "
            "professional for clinical interpretation."
        )
