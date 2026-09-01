import json
import numpy as np
import tensorflow as tf
import gradio as gr


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "fish_species_mobilenetv2.keras"
CLASS_NAMES_PATH = "fish_class_names.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Fish Species Classification System...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully.")
print(f"Number of classes: {len(class_names)}")


# ============================================================
# PREDICTION
# ============================================================

def predict_fish(image):

    if image is None:
        return "Please upload a fish image.", "—"

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array, verbose=0)[0]

    # Best prediction
    predicted_index = np.argmax(predictions)

    predicted_species = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Format output
    species_text = predicted_species.replace("_", " ").title()
    confidence_text = f"{confidence:.2f}%"

    return species_text, confidence_text


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_results():
    return None, "Prediction will appear here", "—"


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """
body {
    background: #f5f7fb;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #667085;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e4e7ec;
}

.footer {
    text-align: center;
    color: #667085;
    margin-top: 20px;
    font-size: 13px;
}
"""


# ============================================================
# INTERFACE
# ============================================================

with gr.Blocks(css=custom_css, title="Fish Species Classifier") as demo:

    gr.Markdown(
        """
        <div class="title">
            # 🐟 Fish Species Classifier
        </div>

        <div class="subtitle">
            AI-powered fish species identification using MobileNetV2
        </div>
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # LEFT SIDE — IMAGE
        # ----------------------------------------------------

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Fish Image",
                height=400
            )

            with gr.Row():

                predict_button = gr.Button(
                    "🔍 Predict Species",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "🗑️ Clear"
                )

        # ----------------------------------------------------
        # RIGHT SIDE — RESULTS
        # ----------------------------------------------------

        with gr.Column(elem_classes="result-box"):

            gr.Markdown("### 🔎 Prediction Result")

            species_output = gr.Textbox(
                label="Predicted Species",
                value="Prediction will appear here",
                interactive=False
            )

            confidence_output = gr.Textbox(
                label="Confidence",
                value="—",
                interactive=False
            )

            gr.Markdown(
                """
                **Model:** MobileNetV2 + Fine-Tuning

                **Supported Classes:** 10 fish species

                Upload a fish image and click **Predict Species**.
                """
            )

    # --------------------------------------------------------
    # BUTTON ACTIONS
    # --------------------------------------------------------

    predict_button.click(
        fn=predict_fish,
        inputs=image_input,
        outputs=[
            species_output,
            confidence_output
        ]
    )

    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[
            image_input,
            species_output,
            confidence_output
        ]
    )

    gr.Markdown(
        """
        <div class="footer">
            Fish Species Classification System • Deep Learning Project
        </div>
        """
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch()