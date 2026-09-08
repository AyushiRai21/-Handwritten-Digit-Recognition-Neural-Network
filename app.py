import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from utils import preprocess_canvas_image

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #555555;
        margin-bottom: 2rem;
    }
    .pred-box {
        background-color: #f0f4f9;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #1E88E5;
    }
    .pred-digit {
        font-size: 4.5rem;
        font-weight: 800;
        color: #1E88E5;
        margin: 0;
        line-height: 1.1;
    }
    .pred-conf {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2E7D32;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load trained models
@st.cache_resource
def load_models():
    models_dict = {}
    if os.path.exists("digit_model.keras"):
        try:
            models_dict["Dense (Baseline MLP)"] = tf.keras.models.load_model("digit_model.keras")
        except Exception as e:
            st.error(f"Error loading dense model: {e}")
            
    if os.path.exists("cnn_model.keras"):
        try:
            models_dict["Convolutional (CNN)"] = tf.keras.models.load_model("cnn_model.keras")
        except Exception as e:
            pass

    return models_dict

# Load model metrics if available
@st.cache_data
def load_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    return None

def main():
    st.markdown("<div class='main-title'>✍️ Handwritten Digit Recognition Neural Network</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Deep Learning Gateway Project • Real-Time Keras Prediction & Model Insights</div>", unsafe_allow_html=True)

    models_dict = load_models()
    metrics = load_metrics()

    if not models_dict:
        st.warning("⚠️ Trained model files not found! Please run `python train_model.py` first to train the model.")

    # Sidebar configuration
    st.sidebar.header("⚙️ Model & Canvas Settings")
    
    selected_model_name = st.sidebar.selectbox(
        "Choose Model Architecture:",
        options=list(models_dict.keys()) if models_dict else ["Dense (Baseline MLP)"]
    )

    stroke_width = st.sidebar.slider("Brush Thickness (px):", min_value=12, max_value=32, value=22, step=2)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Project Overview")
    st.sidebar.markdown("""
    - **Dataset**: MNIST (60,000 train, 10,000 test images)
    - **Resolution**: 28 × 28 grayscale pixels
    - **Preprocessing**: Bounding box crop, aspect-ratio resize, center-of-mass alignment & normalization.
    """)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🖌️ Interactive Canvas & Prediction", "📊 Model Training & Evaluation Insights", "📚 Gateway Learning Notes"])

    with tab1:
        st.markdown("### Draw a single digit (0 - 9) on the canvas below:")
        col_canvas, col_preview, col_prediction = st.columns([2.2, 1.2, 2.5])

        with col_canvas:
            st.write("**Drawing Canvas**")
            
            # Try importing streamlit-drawable-canvas
            try:
                from streamlit_drawable_canvas import st_canvas
                
                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=stroke_width,
                    stroke_color="white",
                    background_color="black",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    return_image_data=True,
                    key="digit_canvas",
                )
                raw_image_data = canvas_result.image_data if canvas_result is not None else None

            except ImportError:
                st.error("`streamlit-drawable-canvas` package is missing. Installing or loading HTML fallback.")
                raw_image_data = None

        # Preprocessing pipeline call
        processed_img, is_blank = preprocess_canvas_image(raw_image_data)

        with col_preview:
            st.write("**Preprocessed (28×28)**")
            if not is_blank:
                # Display 28x28 normalized preview image scaled up for visibility
                st.image(processed_img, width=160, caption="Normalized MNIST-style input", clamp=True)
            else:
                st.info("Canvas is empty. Draw a digit to inspect preprocessed feed.")

        with col_prediction:
            st.write("**Live Prediction & Probabilities**")
            if not is_blank and selected_model_name in models_dict:
                model = models_dict[selected_model_name]
                
                # Format batch input for prediction
                input_tensor = np.expand_dims(processed_img, axis=0)
                if "CNN" in selected_model_name:
                    input_tensor = np.expand_dims(input_tensor, axis=-1)

                probs = model.predict(input_tensor, verbose=0)[0]
                pred_digit = int(np.argmax(probs))
                confidence = float(probs[pred_digit]) * 100

                st.markdown(f"""
                <div class='pred-box'>
                    <div style='font-size:1.1rem; color:#555;'>Predicted Digit</div>
                    <div class='pred-digit'>{pred_digit}</div>
                    <div class='pred-conf'>{confidence:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)

                st.write("**Class Probabilities (0-9):**")
                prob_df = pd.DataFrame({
                    "Digit": [str(i) for i in range(10)],
                    "Probability (%)": probs * 100
                }).set_index("Digit")

                st.bar_chart(prob_df, height=220)
            elif is_blank:
                st.markdown("""
                <div class='pred-box' style='border-color:#ccc;'>
                    <div style='font-size:1.1rem; color:#888;'>Waiting for Drawing</div>
                    <div class='pred-digit' style='color:#bbb;'>?</div>
                    <div style='color:#888;'>Draw digit on the canvas</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📈 Model Training Curves & Error Analysis")
        if metrics:
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Dense Model Test Accuracy", f"{metrics.get('dense_test_accuracy', 0)*100:.2f}%")
            with m_col2:
                st.metric("Dense Model Test Loss", f"{metrics.get('dense_test_loss', 0):.4f}")
            with m_col3:
                if 'cnn_test_accuracy' in metrics:
                    st.metric("CNN Model Test Accuracy", f"{metrics.get('cnn_test_accuracy', 0)*100:.2f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Training & Validation Loss/Accuracy Curves")
            if os.path.exists("loss_accuracy_curves.png"):
                st.image("loss_accuracy_curves.png", use_container_width=True)
            else:
                st.info("Loss curves plot will appear here after training.")

        with c2:
            st.write("##### Confusion Matrix Heatmap")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", use_container_width=True)
            else:
                st.info("Confusion matrix plot will appear here after training.")

        st.markdown("---")
        st.write("##### Common Misclassifications Analysis")
        if os.path.exists("misclassifications.png"):
            st.image("misclassifications.png", use_container_width=True)
            st.caption("Visualizing digits misclassified by the baseline network. Ambiguous handwriting and off-center strokes account for most errors.")

    with tab3:
        st.subheader("🎓 Gateway into Deep Learning: Key Concepts")
        st.markdown("""
        #### 1. Why Loss & Accuracy Curves Matter
        - **Training vs. Validation Loss**: As a neural network trains, its training loss continuously decreases. If the validation loss starts increasing while training loss decreases, the model is **overfitting** (memorizing training examples).
        - **Early Stopping & Regularization**: Dropout layers (e.g. 20% dropout) randomly deactivate neurons during training to prevent the network from depending too heavily on specific node connections.

        #### 2. Bridging the Production Gap: Preprocessing Pipeline
        - Clean training set digits in MNIST are **28×28 pixels**, centered strictly by **center of mass**, with white digits on a black background.
        - Mouse/touch drawings are messy, variable in stroke thickness, off-center, and scaled differently.
        - **Our Pipeline Solution**:
          1. Dynamic bounding box detection.
          2. Aspect-ratio preserving scale to 20×20.
          3. Padding to 28×28.
          4. Center-of-mass shift alignment (`cv2.moments`).

        #### 3. Dense Neural Networks (MLP) vs Convolutional Neural Networks (CNN)
        - **Dense (MLP)**: Flattens the 28×28 matrix into a 784-dimensional vector. It loses 2D spatial relationships between neighboring pixels.
        - **CNN**: Uses 2D convolution filters (`3×3` kernels) and pooling to learn spatial features like edges, curves, loops, and joints invariant to small translations.
        """)

if __name__ == "__main__":
    main()
