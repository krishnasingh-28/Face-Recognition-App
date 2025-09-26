# 🔍 Face Recognition System with Streamlit

A modern, web-based face recognition system built with Streamlit and InsightFace, featuring real-time webcam capture, dark mode support, and an intuitive user interface.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### Core Functionality
- **🎯 High-Accuracy Face Recognition**: Using InsightFace's buffalo_l model
- **📸 Dual Input Methods**: Webcam capture or image upload
- **⚡ Real-time Processing**: Live face detection during capture
- **📊 Similarity Scoring**: Cosine similarity measurement with visual feedback

### User Interface
- **🌙 Dark/Light Mode**: Toggle between themes for comfort
- **📱 Responsive Design**: Works on desktop and mobile browsers
- **🎨 Modern UI**: Gradient backgrounds, smooth animations
- **📈 Visual Metrics**: Progress bars, confidence scores, and result indicators

### Technical Features
- **🔄 Session State Management**: Persistent data across interactions
- **⚙️ Configurable Settings**: Adjustable threshold and capture duration
- **🚀 Performance Optimized**: Model caching and frame skipping options
- **🔒 Secure**: No data storage, all processing in-memory

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Webcam (for live capture feature)
- 4GB+ RAM recommended

## Step 1: Clone the Repository
```bash
git clone https://github.com/krishnasingh-28/Face-Recognition-App.git
cd Face-Recognition-App```


## Step 1.1: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
streamlit>=1.28.0
opencv-python-headless>=4.8.0
insightface>=0.7.3
onnxruntime>=1.15.0
pillow>=10.0.0
numpy>=1.24.0
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run face_recognition_app.py
```
The application will open in your default browser at http://localhost:8501

### Step-by-Step Guide

#### 1. Upload Reference Image
- Click "Browse files" in the Reference Image section
- Select a clear photo with visible face
- Click "Process Reference Image"
- Verify the face is detected (green box)

#### 2. Configure Settings (Optional)
- Adjust similarity threshold (0.0-1.0)
- Set capture duration (1-10 seconds)
- Toggle dark/light mode

#### 3. Verify Face
**Option A: Webcam Capture**
- Select "Automatic Capture"
- Click "Start Webcam Capture"
- Look at camera for specified duration
- System selects best frame automatically

**Option B: Upload Test Image**
- Select "Upload Test Image"
- Choose image file
- Click "Verify"

#### 4. View Results
- ✅ Green box: SAME PERSON (similarity > threshold)
- ❌ Red box: DIFFERENT PERSON (similarity < threshold)
- View detailed metrics and similarity percentage

## 🔧 Configuration

### Model Settings
```python
MODEL_NAME = 'buffalo_l'  # Options: buffalo_l, buffalo_m, buffalo_s
```

### Performance Tuning
```python
PROCESS_EVERY_N_FRAME = 1  # Increase for faster processing
ctx_id = -1  # Set to 0 for GPU acceleration
```

### Threshold Guidelines
- 0.30-0.40: Lenient (more false positives)
- 0.40-0.50: Balanced (recommended)
- 0.50-0.60: Strict (more false negatives)

## 📊 How It Works

### Face Recognition Pipeline

#### Face Detection
- Uses RetinaFace for accurate face detection
- Identifies facial landmarks and bounding boxes

#### Feature Extraction
- Extracts 512-dimensional embedding vector
- Uses ArcFace loss-trained neural network

#### Similarity Calculation
```python
similarity = cosine_similarity(embedding1, embedding2)
```
- Measures angle between face vectors
- Range: 0.0 (different) to 1.0 (identical)

#### Threshold Comparison
- Compares similarity against user-defined threshold
- Returns binary match/no-match decision

### Model Architecture
- **Backbone**: ResNet-based architecture
- **Loss Function**: ArcFace (Additive Angular Margin Loss)
- **Embedding Size**: 512 dimensions
- **Training Data**: MS1MV3 dataset (10M+ images)

## 🎯 Use Cases

- **Security Systems**: Access control and authentication
- **Attendance Systems**: Automated attendance marking
- **Photo Organization**: Group photos by person
- **Identity Verification**: KYC and onboarding processes
- **Event Management**: Guest check-in systems

## ⚠️ Limitations

- Requires good lighting for optimal performance
- Face masks significantly reduce accuracy
- Extreme angles (>45°) may cause detection failure
- Identical twins may produce false positives
- Performance varies with image quality

## 🔒 Privacy & Security

- **No Data Storage**: All processing happens in-memory
- **Local Processing**: No external API calls
- **Session Isolation**: Each user session is independent
- **Temporary Files**: Automatically cleaned up

## 🐛 Troubleshooting

### Common Issues

#### 1. Webcam Not Detected
```bash
# Check webcam availability
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

#### 2. Model Loading Error
```bash
# Clear cache and retry
streamlit cache clear
```

#### 3. Low Performance
- Reduce capture resolution
- Increase PROCESS_EVERY_N_FRAME
- Use GPU acceleration (CUDA)

#### 4. Import Errors
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall insightface
```

## 📈 Performance Metrics

| Operation | CPU Time | GPU Time |
|-----------|----------|----------|
| Model Load | 3-5s | 1-2s |
| Face Detection | 100-200ms | 20-50ms |
| Embedding Extraction | 50-100ms | 10-20ms |
| Similarity Calculation | <1ms | <1ms |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the face recognition models
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision utilities

## 📧 Contact

For questions and support, please open an issue on GitHub or contact:

- **Email**: krishnasingh8404@gmail.com
- **GitHub**: https://github.com/krishnasingh-28

## 🔮 Future Enhancements

- [ ] Multi-face recognition
- [ ] Face recognition from video files
- [ ] Export verification history
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Face clustering feature
- [ ] Mobile app integration

---

Made by Krishna using Streamlit and InsightFace
