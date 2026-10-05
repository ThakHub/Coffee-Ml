import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Coffee Bean Classification",
    page_icon="☕",
    layout="centered"
)

# =========================================================
# Title
# =========================================================
st.header("☕ Coffee Bean Image Classification")
st.caption("อัปโหลดภาพเมล็ดกาแฟ แล้วให้โมเดลทำนายสายพันธุ์ (Arabica / Excelsa / Liberica / Robusta)")

st.divider()

# =========================================================
# Class names (ต้องเรียงตาม class_indices ตอนเทรน)
# จากโค้ดเทรน: {'Arabica': 0, 'Excelsa': 1, 'Liberica': 2, 'Robusta': 3}
# =========================================================
CLASS_NAMES = ["Arabica", "Excelsa", "Liberica", "Robusta"]
CLASS_EMOJI = {
    "Arabica": "🟢",
    "Excelsa": "🟡",
    "Liberica": "🟤",
    "Robusta": "🔴"
}

MODEL_PATH = "model/DenseNet121_best.keras"   # เปลี่ยนเป็น .h5 ถ้าใช้ไฟล์นั้น

# =========================================================
# Load Model
# =========================================================
@st.cache_resource
def load_classification_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_classification_model()
except Exception as e:
    st.error("❌ ไม่สามารถโหลด Model ได้")
    st.info(
        "กรุณาตรวจสอบว่าไฟล์ model อยู่ในโฟลเดอร์ "
        "`model/` และชื่อไฟล์ถูกต้อง"
    )
    st.code(MODEL_PATH)
    st.exception(e)
    st.stop()

# =========================================================
# Upload Image
# =========================================================
st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "เลือกภาพเมล็ดกาแฟ (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    with col2:
        st.info(
            f"""
            **File:** {uploaded_file.name}  
            **Size:** {image.size[0]} × {image.size[1]} pixels
            """
        )

    st.divider()

    # =====================================================
    # Image Preprocessing (ต้องตรงกับตอนเทรน)
    # =====================================================
    IMG_SIZE = 224
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image_resized).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)   # (1, 224, 224, 3)

    # =====================================================
    # Prediction
    # =====================================================
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        with st.spinner("Model is analyzing the image..."):
            prediction = model.predict(image_array, verbose=0)[0]

        predicted_idx = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(prediction[predicted_idx])
        confidence_percent = confidence * 100

        # =================================================
        # Prediction Result
        # =================================================
        st.subheader("🎯 Prediction Result")

        emoji = CLASS_EMOJI.get(predicted_class, "☕")
        st.success(f"{emoji} The model predicts: **{predicted_class}**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", predicted_class)
        with col2:
            st.metric("Confidence", f"{confidence_percent:.2f}%")

        # Progress bar
        st.write("**Confidence Level**")
        st.progress(min(int(confidence_percent), 100))

        # =================================================
        # Probability Details
        # =================================================
        with st.expander("📊 Show Prediction Details"):
            st.write("**Probability of each class**")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = prediction[i] * 100
                st.write(f"{CLASS_EMOJI[class_name]} **{class_name}:** {prob:.2f}%")
                st.progress(min(int(prob), 100))

            st.write(f"Model input size: **{IMG_SIZE} × {IMG_SIZE} × 3**")
            st.write(f"Model file: `{MODEL_PATH}`")

else:
    st.info("👆 กรุณาอัปโหลดภาพเมล็ดกาแฟเพื่อเริ่มทำนาย")

# =========================================================
# Footer Information
# =========================================================
st.divider()

with st.expander("ℹ️ About this application"):
    st.markdown(
        """
        แอปพลิเคชันนี้ใช้โมเดล **DenseNet121** ที่ผ่านการ Transfer Learning 
        จาก ImageNet เพื่อจำแนกสายพันธุ์เมล็ดกาแฟ 4 ชนิด

        **Classes**
        - 🟢 Arabica
        - 🟡 Excelsa
        - 🟤 Liberica
        - 🔴 Robusta

        **Input image**
        - ปรับขนาดเป็น 224 × 224 pixels
        - RGB
        - Normalized ให้อยู่ในช่วง 0–1

        **Model**
        - Keras / TensorFlow
        - Softmax output (4 classes)
        - Best Test Accuracy: **70.81%**
        """
    )