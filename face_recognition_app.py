import streamlit as st
import cv2
import numpy as np
import insightface
import time
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
class Config:
    MODEL_NAME = 'buffalo_l'
    CAPTURE_DURATION_SECONDS = 3
    SIMILARITY_THRESHOLD = 0.40
    PROCESS_EVERY_N_FRAME = 1

# --- Initialize Session State ---
if 'app' not in st.session_state:
    st.session_state.app = None
    st.session_state.reference_embedding = None
    st.session_state.reference_image = None
    st.session_state.capture_complete = False
    st.session_state.verification_result = None
    st.session_state.dark_mode = False

# --- Theme Management ---
def get_theme_styles():
    """Return CSS styles based on current theme."""
    if st.session_state.dark_mode:
        # Dark mode styles
        return """
        <style>
        /* Dark Mode Styles */
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .main-header {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        
        .success-box {
            background-color: #1a4d2e;
            color: #4ade80;
            border: 2px solid #22c55e;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
        }
        
        .failure-box {
            background-color: #4d1a1a;
            color: #f87171;
            border: 2px solid #dc2626;
            box-shadow: 0 0 20px rgba(220, 38, 38, 0.3);
        }
        
        .info-box {
            background-color: #1e3a5f;
            color: #60a5fa;
            border: 1px solid #3b82f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .theme-toggle {
            position: fixed;
            top: 14px;
            right: 50px;
            z-index: 999999;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 30px;
            padding: 8px 20px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Dark mode for Streamlit components */
        .stSelectbox > div > div {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        .uploadedFile {
            background-color: #2d2d2d !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        div[data-testid="stSidebar"] {
            background-color: #0f0f0f;
            border-right: 1px solid #333;
        }
        
        div[data-testid="stSidebar"] .stMarkdown {
            color: #ffffff;
        }
        
        .metric-container {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #444;
            margin: 10px 0;
        }
        
        /* Progress bar dark mode */
        .stProgress > div > div > div {
            background-color: #667eea;
        }
        
        /* Footer dark mode */
        .footer-text {
            text-align: center;
            color: #888;
            padding: 20px;
            border-top: 1px solid #333;
            margin-top: 50px;
        }
        </style>
        """
    else:
        # Light mode styles
        return """
        <style>
        /* Light Mode Styles */
        .main-header {
            text-align: center;
            color: #2E86AB;
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .result-box {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        
        .success-box {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #28a745;
            box-shadow: 0 0 15px rgba(40, 167, 69, 0.2);
        }
        
        .failure-box {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #dc3545;
            box-shadow: 0 0 15px rgba(220, 53, 69, 0.2);
        }
        
        .info-box {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .theme-toggle {
            position: fixed;
            top: 14px;
            right: 50px;
            z-index: 999999;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: none;
            border-radius: 30px;
            padding: 8px 20px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .metric-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin: 10px 0;
        }
        
        /* Footer light mode */
        .footer-text {
            text-align: center;
            color: #666;
            padding: 20px;
            border-top: 1px solid #dee2e6;
            margin-top: 50px;
        }
        </style>
        """

# --- Apply Theme ---
st.markdown(get_theme_styles(), unsafe_allow_html=True)

# --- Theme Toggle Button ---
def create_theme_toggle():
    """Create a theme toggle button using HTML/JavaScript."""
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    theme_text = "Dark Mode" if not st.session_state.dark_mode else "Light Mode"
    
    st.markdown(f"""
        <button class="theme-toggle" onclick="window.location.reload()">
            {theme_icon} {theme_text}
        </button>
    """, unsafe_allow_html=True)

# --- Core Functions ---
@st.cache_resource
def load_model():
    """Load and cache the face recognition model."""
    with st.spinner("Loading face recognition model..."):
        app = insightface.app.FaceAnalysis(name=Config.MODEL_NAME)
        app.prepare(ctx_id=-1)  # CPU mode
    return app

def extract_face_embedding(app, image):
    """Extract face embedding from an image."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in BGR format for OpenCV
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    faces = app.get(image)
    
    if not faces:
        return None, None
    
    # Get the largest face
    largest_face = max(faces, key=lambda face: 
                      (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    
    # Draw bounding box on image
    bbox = largest_face.bbox.astype(int)
    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Add confidence score
    conf_text = f"Confidence: {largest_face.det_score:.2f}"
    cv2.putText(annotated_image, conf_text, (bbox[0], bbox[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return largest_face.embedding, annotated_image

def calculate_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def capture_from_webcam(app, duration=3):
    """Capture best face from webcam with progress bar."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Please check your camera connection.")
        return None, None
    
    best_face = None
    best_frame = None
    best_quality_score = -1.0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        progress = min(elapsed_time / duration, 1.0)
        progress_bar.progress(progress)
        
        remaining_time = max(0, int(duration - elapsed_time))
        status_text.text(f"⏱️ Capturing... Time remaining: {remaining_time}s")
        
        # Process every N-th frame
        if frame_count % Config.PROCESS_EVERY_N_FRAME == 0:
            faces = app.get(frame)
            
            if faces:
                for face in faces:
                    bbox = face.bbox
                    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    quality_score = face.det_score * face_area
                    
                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_frame = frame.copy()
                        best_face = face
                
                # Draw bounding box on current frame
                display_frame = frame.copy()
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Show the frame with face detection
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text("✅ Capture complete!")
    
    if best_face is not None and best_frame is not None:
        # Convert best frame to RGB and add bounding box
        best_frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        bbox = best_face.bbox.astype(int)
        cv2.rectangle(best_frame_rgb, (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), (0, 255, 0), 3)
        return best_face.embedding, best_frame_rgb
    
    return None, None

# --- Main App ---
def main():
    # Theme toggle in sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark" if not st.session_state.dark_mode else "☀️ Light", 
                        key="theme_btn",
                        use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        with col2:
            mode_text = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
            st.markdown(f"<p style='padding: 5px;'>{mode_text}</p>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Face Recognition System</h1>', unsafe_allow_html=True)
    
    # Load model
    if st.session_state.app is None:
        st.session_state.app = load_model()
        st.success("✅ Model loaded successfully!")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Threshold slider
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=Config.SIMILARITY_THRESHOLD,
            step=0.01,
            help="Higher threshold = stricter matching"
        )
        Config.SIMILARITY_THRESHOLD = threshold
        
        # Capture duration
        capture_duration = st.slider(
            "Capture Duration (seconds)",
            min_value=1,
            max_value=10,
            value=Config.CAPTURE_DURATION_SECONDS,
            help="Duration for webcam capture"
        )
        Config.CAPTURE_DURATION_SECONDS = capture_duration
        
        st.markdown("---")
        st.markdown("### 📊 Current Settings")
        
        # Settings display with theme-aware styling
        settings_style = "background-color: #2d2d2d; color: white;" if st.session_state.dark_mode else "background-color: #f0f2f6;"
        st.markdown(f"""
        <div style='padding: 15px; border-radius: 10px; {settings_style}'>
            <p><strong>Model:</strong> {Config.MODEL_NAME}</p>
            <p><strong>Threshold:</strong> {threshold:.2f}</p>
            <p><strong>Capture Time:</strong> {capture_duration}s</p>
            <p><strong>Theme:</strong> {'Dark' if st.session_state.dark_mode else 'Light'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📸 Reference Image")
        
        # Upload reference image
        uploaded_file = st.file_uploader(
            "Choose a reference image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo with face visible"
        )
        
        if uploaded_file is not None:
            # Load and display reference image
            reference_image = Image.open(uploaded_file)
            st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)
            
            # Extract embedding
            if st.button("🔍 Process Reference Image", key="process_ref"):
                with st.spinner("Extracting face features..."):
                    embedding, annotated_img = extract_face_embedding(
                        st.session_state.app, reference_image
                    )
                    
                    if embedding is not None:
                        st.session_state.reference_embedding = embedding
                        st.session_state.reference_image = annotated_img
                        st.success("✅ Reference face processed successfully!")
                        st.image(annotated_img, caption="Detected Face", use_column_width=True)
                    else:
                        st.error("❌ No face detected in the reference image!")
    
    with col2:
        st.header("📹 Live Capture")
        
        if st.session_state.reference_embedding is not None:
            st.markdown('<div class="info-box">✅ Reference image loaded. Ready for verification!</div>', 
                       unsafe_allow_html=True)
            
            # Webcam capture options
            capture_method = st.radio(
                "Select capture method:",
                ["🎥 Automatic Capture", "📤 Upload Test Image"],
                horizontal=True
            )
            
            if capture_method == "🎥 Automatic Capture":
                if st.button("📷 Start Webcam Capture", key="capture", use_container_width=True):
                    st.markdown("### Live Feed")
                    webcam_embedding, captured_frame = capture_from_webcam(
                        st.session_state.app, 
                        Config.CAPTURE_DURATION_SECONDS
                    )
                    
                    if webcam_embedding is not None:
                        # Calculate similarity
                        similarity = calculate_similarity(
                            st.session_state.reference_embedding, 
                            webcam_embedding
                        )
                        
                        # Display captured frame
                        st.image(captured_frame, caption="Best Captured Frame", use_column_width=True)
                        
                        # Show result
                        st.markdown("---")
                        if similarity > Config.SIMILARITY_THRESHOLD:
                            st.markdown(
                                f'<div class="result-box success-box">✅ SAME PERSON<br>Similarity: {similarity:.4f}</div>',
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        else:
                            st.markdown(
                                f'<div class="result-box failure-box">❌ DIFFERENT PERSON<br>Similarity: {similarity:.4f}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Show similarity meter
                        st.markdown("### 📊 Similarity Analysis")
                        
                        # Progress bar with custom color
                        progress_color = "green" if similarity > Config.SIMILARITY_THRESHOLD else "red"
                        st.markdown(f"""
                        <div style='margin: 20px 0;'>
                            <div style='background-color: #e0e0e0; border-radius: 10px; overflow: hidden;'>
                                <div style='width: {similarity*100}%; background-color: {progress_color}; 
                                          height: 30px; display: flex; align-items: center; 
                                          justify-content: center; color: white; font-weight: bold;'>
                                    {similarity:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional metrics
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Similarity", f"{similarity:.4f}")
                        with col_b:
                            st.metric("Threshold", f"{Config.SIMILARITY_THRESHOLD:.2f}")
                        with col_c:
                            status = "✅ Match" if similarity > Config.SIMILARITY_THRESHOLD else "❌ No Match"
                            st.metric("Status", status)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ No face detected during capture!")
            
            else:  # Upload Test Image
                test_file = st.file_uploader(
                    "Choose a test image",
                    type=['jpg', 'jpeg', 'png'],
                    key="test_upload",
                    help="Upload an image to verify against reference"
                )
                
                if test_file is not None:
                    test_image = Image.open(test_file)
                    st.image(test_image, caption="Test Image", use_column_width=True)
                    
                    if st.button("🔍 Verify", key="verify", use_container_width=True):
                        with st.spinner("Verifying..."):
                            test_embedding, annotated_test = extract_face_embedding(
                                st.session_state.app, test_image
                            )
                            
                            if test_embedding is not None:
                                similarity = calculate_similarity(
                                    st.session_state.reference_embedding,
                                    test_embedding
                                )
                                
                                st.image(annotated_test, caption="Detected Face in Test Image", use_column_width=True)
                                
                                # Show result
                                st.markdown("---")
                                if similarity > Config.SIMILARITY_THRESHOLD:
                                    st.markdown(
                                        f'<div class="result-box success-box">✅ SAME PERSON<br>Similarity: {similarity:.4f}</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.balloons()
                                else:
                                    st.markdown(
                                        f'<div class="result-box failure-box">❌ DIFFERENT PERSON<br>Similarity: {similarity:.4f}</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Show similarity meter with custom styling
                                st.markdown("### 📊 Similarity Analysis")
                                progress_color = "green" if similarity > Config.SIMILARITY_THRESHOLD else "red"
                                st.markdown(f"""
                                <div style='margin: 20px 0;'>
                                    <div style='background-color: #e0e0e0; border-radius: 10px; overflow: hidden;'>
                                        <div style='width: {similarity*100}%; background-color: {progress_color}; 
                                                height: 30px; display: flex; align-items: center; 
                                                justify-content: center; color: white; font-weight: bold;'>
                                            {similarity:.1%}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error("❌ No face detected in test image!")
        else:
            st.warning("⚠️ Please upload and process a reference image first!")
    
    # Footer
    st.markdown("---")
    footer_class = "footer-text"
    st.markdown(
        f"""
        <div class='{footer_class}'>
            <p>🔍 Face Recognition System using InsightFace</p>
            <p>Built by Krishna using Streamlit & OpenCV</p>
            <p>Theme: {'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()