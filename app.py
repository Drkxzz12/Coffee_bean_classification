import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Bean Quality Classifier",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better UI ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #8B4513;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #5D4037;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #ddd;
        margin: 1rem 0;
    }
    .prediction-good {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #155724;
    }
    .prediction-bad {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        border: 2px solid #dc3545;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #721c24;
    }
    .prediction-uncertain {
        background: linear-gradient(135deg, #fff3cd 0%, #fce4ec 100%);
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #856404;
    }
    .confidence-metric {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 4px solid #007bff;
    }
    .metric-good {
        border-left-color: #28a745;
    }
    .metric-bad {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Model Loading ---
@st.cache_resource
def load_models():
    """Load the two trained models from disk with error handling."""
    models = {}
    model_files = {
        "Custom CNN": "custom_model.h5",
        "Transfer Learning": "CoffeeBeanbest_model.h5"
    }
    
    for model_name, filename in model_files.items():
        try:
            model = tf.keras.models.load_model(filename)
            models[model_name] = model
            st.sidebar.success(f"✅ {model_name} loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load {model_name}: {str(e)}")
            models[model_name] = None
    
    return models

# --- Enhanced Image Preprocessing ---
def preprocess_image(image, target_size=(224, 224), enhance_contrast=False):
    """Enhanced preprocessing with optional image enhancement."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Optional contrast enhancement
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast by 20%
    
    # Resize image
    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    image_array = np.asarray(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype(np.float32) / 255.0
    
    return image_array

# --- Enhanced Prediction Function ---
def make_enhanced_prediction(model, processed_image, model_name):
    """Make prediction with confidence intervals and uncertainty estimation."""
    try:
        # Get raw prediction
        raw_prediction = model.predict(processed_image, verbose=0)[0]
        
        # Handle different output formats
        if len(raw_prediction) == 1:
            # Binary classification with sigmoid output
            confidence_good = float(raw_prediction[0]) * 100
            confidence_bad = (1 - float(raw_prediction[0])) * 100
        else:
            # Multi-class output (softmax)
            confidence_bad = float(raw_prediction[0]) * 100
            confidence_good = float(raw_prediction[1]) * 100
        
        # Determine prediction with more nuanced thresholds
        if confidence_good > 70:
            prediction = "Good Bean"
            certainty = "High"
            color = "good"
        elif confidence_good > 55:
            prediction = "Good Bean"
            certainty = "Medium"
            color = "good"
        elif confidence_bad > 70:
            prediction = "Bad Bean"
            certainty = "High"
            color = "bad"
        elif confidence_bad > 55:
            prediction = "Bad Bean"
            certainty = "Medium"
            color = "bad"
        else:
            prediction = "Uncertain"
            certainty = "Low"
            color = "uncertain"
        
        return {
            "prediction": prediction,
            "confidence_good": confidence_good,
            "confidence_bad": confidence_bad,
            "certainty": certainty,
            "color": color,
            "raw_output": raw_prediction
        }
    
    except Exception as e:
        st.error(f"Error making prediction with {model_name}: {str(e)}")
        return None

# --- Confidence Visualization ---
def display_confidence_metrics(result, model_name):
    """Display confidence metrics in a clean format."""
    st.markdown(f"**📊 {model_name} Confidence:**")
    
    # Good bean confidence
    st.markdown(f"""
    <div class="confidence-metric metric-good">
        🟢 Good Bean: <strong>{result['confidence_good']:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Bad bean confidence
    st.markdown(f"""
    <div class="confidence-metric metric-bad">
        🔴 Bad Bean: <strong>{result['confidence_bad']:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bars
    st.progress(result['confidence_good'] / 100, text=f"Good Bean: {result['confidence_good']:.1f}%")
    st.progress(result['confidence_bad'] / 100, text=f"Bad Bean: {result['confidence_bad']:.1f}%")

# --- Main App ---
def main():
    # Header
    st.markdown('<h1 class="main-header">☕ Coffee Bean Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-powered coffee bean quality assessment with model comparison</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    # Sidebar controls
    st.sidebar.header("🛠️ Settings")
    enhance_contrast = st.sidebar.checkbox("Enhance image contrast", value=False, help="Apply contrast enhancement to potentially improve prediction accuracy")
    confidence_threshold = st.sidebar.slider("Confidence threshold for 'certain' predictions", 50, 90, 70, help="Adjust the minimum confidence required for high-certainty predictions")
    show_raw_data = st.sidebar.checkbox("Show raw model outputs", value=False, help="Display technical details about model predictions")
    
    # Main content area
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "📸 Upload your coffee bean image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Supported formats: JPG, PNG, BMP, TIFF"
        )
    
    with col_info:
        with st.expander("ℹ️ How to get best results"):
            st.write("""
            **For optimal predictions:**
            - Use clear, well-lit images
            - Ensure beans are clearly visible
            - Avoid blurry or dark images
            - Single bean or small groups work best
            - Try contrast enhancement for low-light images
            """)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Image info
        st.subheader("📷 Uploaded Image")
        col_img, col_details = st.columns([2, 1])
        
        with col_img:
            st.image(image, caption="Original Image", use_column_width=True)
        
        with col_details:
            st.write("**Image Details:**")
            st.write(f"- Size: {image.size[0]} × {image.size[1]} pixels")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Format: {image.format}")
            
            # Show preprocessed image if contrast enhancement is enabled
            if enhance_contrast:
                enhanced = ImageEnhance.Contrast(image).enhance(1.2)
                st.image(enhanced, caption="Enhanced Image", use_column_width=True)
        
        st.divider()
        
        # Model predictions
        st.subheader("🤖 Model Predictions")
        
        # Preprocess image
        processed_image = preprocess_image(image, enhance_contrast=enhance_contrast)
        
        # Get predictions from both models
        results = {}
        prediction_cols = st.columns(2)
        
        for i, (model_name, model) in enumerate(models.items()):
            with prediction_cols[i]:
                st.markdown(f"### {model_name}")
                
                if model is not None:
                    with st.spinner(f"🔄 Analyzing with {model_name}..."):
                        time.sleep(0.5)  # Small delay for better UX
                        result = make_enhanced_prediction(model, processed_image, model_name)
                        results[model_name] = result
                    
                    if result:
                        # Prediction card
                        if result['color'] == 'good':
                            st.markdown(f'<div class="prediction-good">🟢 {result["prediction"]}<br>Confidence: {result["confidence_good"]:.1f}%<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)
                        elif result['color'] == 'bad':
                            st.markdown(f'<div class="prediction-bad">🔴 {result["prediction"]}<br>Confidence: {result["confidence_bad"]:.1f}%<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="prediction-uncertain">🟡 {result["prediction"]}<br>Certainty: {result["certainty"]}</div>', unsafe_allow_html=True)
                        
                        # Detailed confidence display
                        with st.expander("📊 View Detailed Confidence"):
                            display_confidence_metrics(result, model_name)
                        
                        # Raw data (optional)
                        if show_raw_data:
                            with st.expander("🔧 Raw Model Output"):
                                st.code(str(result['raw_output']))
                
                else:
                    st.error(f"❌ {model_name} not available")
        
        # Comparative analysis
        valid_results = {k: v for k, v in results.items() if v}
        if len(valid_results) > 1:
            st.divider()
            st.subheader("🔍 Model Comparison")
            
            # Agreement analysis
            predictions = [r['prediction'] for r in valid_results.values()]
            if len(set(predictions)) == 1:
                st.success(f"✅ **Model Agreement**: Both models agree - {predictions[0]}")
            else:
                st.warning("⚠️ **Model Disagreement**: Models have different predictions. Consider the confidence levels.")
            
            # Show detailed comparison in columns
            comp_cols = st.columns(len(valid_results))
            for i, (model_name, result) in enumerate(valid_results.items()):
                with comp_cols[i]:
                    st.write(f"**{model_name}**")
                    st.write(f"Prediction: {result['prediction']}")
                    st.write(f"Max Confidence: {max(result['confidence_good'], result['confidence_bad']):.1f}%")
                    st.write(f"Certainty: {result['certainty']}")
            
            # Simple confidence chart using metrics
            st.subheader("📈 Confidence Comparison")
            chart_cols = st.columns(2)
            
            for i, (model_name, result) in enumerate(valid_results.items()):
                with chart_cols[i]:
                    st.write(f"**{model_name}**")
                    st.metric("Good Bean", f"{result['confidence_good']:.1f}%", 
                            delta=f"{result['confidence_good'] - 50:.1f}%" if result['confidence_good'] != 50 else None)
                    st.metric("Bad Bean", f"{result['confidence_bad']:.1f}%", 
                            delta=f"{result['confidence_bad'] - 50:.1f}%" if result['confidence_bad'] != 50 else None)
    
    else:
        # Welcome message and instructions
        st.info("👆 Upload a coffee bean image to start the analysis")
        
        # Sample results or tips
        st.subheader("🎯 About This Classifier")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Custom CNN Model:**
            - Lightweight architecture
            - Fast inference
            - Optimized for coffee beans
            - Good for real-time applications
            """)
        
        with col2:
            st.write("""
            **ResNet50 Transfer Learning Model:**
            - Based on ResNet50 architecture
            - Pre-trained on ImageNet dataset
            - 50 layers deep neural network
            - Excellent feature extraction capabilities
            - Uses residual connections for better training
            - ImageNet-optimized preprocessing
            """)

if __name__ == "__main__":
    main()