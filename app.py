import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Fish Species AI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed"
)


MODEL_PATH = "fish_species_mobilenetv2.keras"
CLASS_NAMES_PATH = "fish_class_names.json"
IMAGE_SIZE = (224, 224)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(14,165,233,0.10), transparent 30%),
        #f7f9fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 25px 10px 35px 10px;
}

.hero-icon {
    font-size: 58px;
    margin-bottom: 5px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1.5px;
}

.hero-title span {
    color: #1683d8;
}

.hero-subtitle {
    color: #667085;
    font-size: 17px;
    margin-top: 10px;
}

.badges {
    text-align: center;
    margin-top: 18px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    margin: 4px;
    border-radius: 30px;
    background: #eaf4ff;
    color: #1267a5;
    font-size: 13px;
    font-weight: 600;
}

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    height: 100%;
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 18px;
}

.result-card {
    background: linear-gradient(145deg, #ffffff, #f2f8ff);
    border: 1px solid #d9eafd;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.07);
}

.result-label {
    color: #667085;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.species {
    color: #1267a5;
    font-size: 30px;
    font-weight: 800;
    margin: 8px 0 15px 0;
}

.confidence {
    font-size: 25px;
    font-weight: 700;
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 35px;
    margin-bottom: 15px;
}

.info-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.info-number {
    font-size: 27px;
    font-weight: 800;
    color: #1683d8;
}

.info-label {
    color: #667085;
    font-size: 13px;
    margin-top: 3px;
}

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 13px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
}

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 8px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 14px;
}

.top-prediction {
    background: #f8fafc;
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error("Unable to load the AI classification system.")
    st.exception(error)
    st.stop()


def predict_fish(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    predicted_index = int(top_indices[0])
    predicted_species = class_names[predicted_index]

    confidence = float(predictions[predicted_index] * 100)

    top_predictions = []

    for index in top_indices:
        species = class_names[int(index)]
        probability = float(predictions[int(index)] * 100)
        top_predictions.append((species, probability))

    return predicted_species, confidence, top_predictions


st.markdown("""
<div class="hero">
    <div class="hero-icon">🐟</div>
    <h1 class="hero-title">Fish Species <span>AI Classifier</span></h1>
    <p class="hero-subtitle">
        Identify fish species instantly using Deep Learning and MobileNetV2
    </p>
    <div class="badges">
        <span class="badge">🤖 AI Powered</span>
        <span class="badge">🧠 MobileNetV2</span>
        <span class="badge">🎯 10 Species</span>
        <span class="badge">⚡ Fast Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)


metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">10</div>
        <div class="info-label">Supported Species</div>
    </div>
    """, unsafe_allow_html=True)

with metric2:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">224×224</div>
        <div class="info-label">Input Image Size</div>
    </div>
    """, unsafe_allow_html=True)

with metric3:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">CNN</div>
        <div class="info-label">Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

with metric4:
    st.markdown("""
    <div class="info-box">
        <div class="info-number">AI</div>
        <div class="info-label">Image Classification</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="section-title">🔬 Fish Identification</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")


with left:

    st.markdown("""
    <div class="card">
        <div class="card-title">📤 Upload Your Fish Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Fish Image",
            use_container_width=True
        )

        predict = st.button(
            "🔍 Predict Fish Species",
            type="primary"
        )

        clear = st.button(
            "🔄 Clear",
            type="secondary"
        )

        if clear:
            for key in [
                "predicted_species",
                "confidence",
                "top_predictions"
            ]:
                st.session_state.pop(key, None)

            st.rerun()

        if predict:

            with st.spinner("AI is analyzing the image..."):

                species, confidence, top_predictions = predict_fish(image)

                st.session_state["predicted_species"] = species
                st.session_state["confidence"] = confidence
                st.session_state["top_predictions"] = top_predictions


with right:

    if "predicted_species" in st.session_state:

        species = st.session_state["predicted_species"]
        confidence = st.session_state["confidence"]
        top_predictions = st.session_state["top_predictions"]

        display_species = species.replace("_", " ").title()

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Species</div>
            <div class="species">🐟 {display_species}</div>
            <div class="result-label">Confidence</div>
            <div class="confidence">{confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(
            min(max(float(confidence) / 100.0, 0.0), 1.0)
        )

        st.markdown(
            '<div class="section-title">📊 Top 3 Predictions</div>',
            unsafe_allow_html=True
        )

        for rank, (species_name, probability) in enumerate(
            top_predictions,
            start=1
        ):

            display_name = species_name.replace("_", " ").title()

            st.markdown(
                f"""
                <div class="top-prediction">
                    <strong>#{rank} {display_name}</strong>
                    <span style="float:right;">
                        {probability:.2f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(max(float(probability) / 100.0, 0.0), 1.0)
            )

    else:

        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Prediction Result</div>
            <p style="color:#667085;">
                Upload a fish image on the left and click
                <b>Predict Fish Species</b>.
            </p>
            <br>
            <p>
                The AI model will analyze the image and identify
                the most likely fish species together with its
                prediction confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">🐠 Supported Fish Species</div>',
    unsafe_allow_html=True
)

species_columns = st.columns(2)

for index, species in enumerate(class_names):

    display_name = species.replace("_", " ").title()

    with species_columns[index % 2]:

        st.markdown(
            f"""
            <div class="top-prediction">
                🐟 <strong>{display_name}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("""
<div class="footer">
    <b>Fish Species AI Classifier</b><br>
    Deep Learning Project • MobileNetV2 + Fine-Tuning<br>
    Built for intelligent fish species identification
</div>
""", unsafe_allow_html=True)