import cv2
import numpy as np
import insightface
from matplotlib import pyplot as plt
import time

# --- Configuration ---
class Config:
    MODEL_NAME = 'buffalo_l'
    # CORRECTED: Filename now matches the actual file with underscores.
    REFERENCE_IMAGE_PATH = "passport_size_photo.jpg" 
    CAPTURE_DURATION_SECONDS = 3
    SIMILARITY_THRESHOLD = 0.40  # Tuned threshold for buffalo_l
    PROCESS_EVERY_N_FRAME = 1   # For performance optimization

# --- Core Functions ---

def load_reference_face(app, image_path):
    """Loads the reference image and extracts the face embedding."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"❌ Error: Reference image not found at '{image_path}'")
        return None, None
    
    faces = app.get(img)
    if not faces:
        print(f"❌ Error: No face detected in the reference image '{image_path}'")
        return None, None
    
    # Using the largest face in the reference image
    largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    return largest_face.embedding, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# IMPROVED: Removed the unused 'reference_emb' parameter for better clarity.
def capture_best_face(app):
    """Captures video from the webcam and finds the best quality face."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        return None, None

    print("✅ Starting webcam... Please look at the camera.")

    best_frame_for_match = None
    best_face_for_match = None
    best_quality_score = -1.0
    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < Config.CAPTURE_DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        frame_count += 1
        
        # --- Performance Optimization: Process every N-th frame ---
        if frame_count % Config.PROCESS_EVERY_N_FRAME == 0:
            faces_in_frame = app.get(frame)
            
            if faces_in_frame:
                # Find the best quality face in the current frame
                for face in faces_in_frame:
                    bbox = face.bbox
                    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    # Combined score using detection confidence and face size
                    quality_score = face.det_score * face_area 

                    # Update the overall best face found so far
                    if quality_score > best_quality_score:
                        best_quality_score = quality_score
                        best_frame_for_match = frame.copy()
                        best_face_for_match = face
                
                # Draw a box around the largest face in the *current* frame for feedback
                best_in_current = max(faces_in_frame, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                bbox = best_in_current.bbox.astype(int)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # --- Real-time Feedback on the video stream ---
        remaining_time = max(0, int(Config.CAPTURE_DURATION_SECONDS - (time.time() - start_time)))
        feedback_text = f"Time left: {remaining_time}s"
        cv2.putText(display_frame, feedback_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Capturing Best Frame...", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("✅ Capture finished.")
    cap.release()
    cv2.destroyAllWindows()
    
    return best_face_for_match, best_frame_for_match


def main():
    """Main function to run the face verification process."""
    # 1. Initialize Model
    app = insightface.app.FaceAnalysis(name=Config.MODEL_NAME)
    # IMPROVED: Use ctx_id=-1 for CPU to ensure it runs on all computers.
    # Change to ctx_id=0 if you have a compatible GPU.
    app.prepare(ctx_id=-1) 

    # 2. Load Reference Face
    ref_emb, ref_img_rgb = load_reference_face(app, Config.REFERENCE_IMAGE_PATH)
    if ref_emb is None:
        return

    # 3. Capture from Webcam and Find Best Face
    webcam_face, best_frame = capture_best_face(app)

    # 4. Compare and Display Results
    if webcam_face is None or best_frame is None:
        print("❌ No face was detected from the webcam feed.")
        return

    webcam_emb = webcam_face.embedding
    # Calculate cosine similarity
    sim = np.dot(ref_emb, webcam_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(webcam_emb))
    
    if sim > Config.SIMILARITY_THRESHOLD:
        result_text = f"✅ SAME PERSON\nCosine Similarity: {sim:.4f}"
        color = "green"
    else:
        result_text = f"❌ DIFFERENT PERSON\nCosine Similarity: {sim:.4f}"
        color = "red"
    
    # --- Display Final Comparison ---
    webcam_img_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(ref_img_rgb)
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(webcam_img_rgb)
    plt.title("Best Frame from Webcam")
    plt.axis("off")

    plt.suptitle(result_text, fontsize=14, color=color)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()