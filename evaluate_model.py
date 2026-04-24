import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
from datetime import datetime

# Import the same constants and functions from drowsiness_detector
# Eye landmarks indices for FaceMesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 362, 380, 374]

# Mouth landmarks indices for FaceMesh
MOUTH_INDICES = [61, 291, 78, 308, 13, 14]

# Head pose landmarks indices
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def calculate_ear(eye_landmarks):
    if len(eye_landmarks) < 6:
        return 0
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    return ear

def calculate_mar(mouth_landmarks):
    if len(mouth_landmarks) < 6:
        return 0
    vertical_1 = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    vertical_2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[5])
    horizontal = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
    mar = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    return mar

def get_eye_landmarks(landmarks, indices):
    eye_points = []
    for idx in indices:
        point = landmarks[idx]
        eye_points.append(np.array([point.x, point.y]))
    return eye_points

def get_mouth_landmarks(landmarks, indices):
    mouth_points = []
    for idx in indices:
        point = landmarks[idx]
        mouth_points.append(np.array([point.x, point.y]))
    return mouth_points

def get_head_pose_landmarks(landmarks, indices, width, height):
    image_points = []
    for idx in indices:
        point = landmarks[idx]
        x = point.x * width
        y = point.y * height
        image_points.append([x, y])
    return np.array(image_points, dtype=np.float64)

def calculate_head_pose(image_points, width, height):
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0, 0, 0
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    
    return pitch, yaw, roll

def main():
    print("=" * 60)
    print("DROWSINESS DETECTION MODEL EVALUATION")
    print("=" * 60)
    
    # Load existing data if available
    existing_predictions = []
    existing_ground_truth = []
    existing_samples = []
    
    try:
        with open('evaluation_data.json', 'r') as f:
            data = json.load(f)
            existing_samples = data.get('samples', [])
            # Reconstruct predictions and ground truth from samples
            for sample in existing_samples:
                existing_predictions.append(1 if sample['predicted'] == 'Drowsy' else 0)
                existing_ground_truth.append(1 if sample['actual'] == 'Drowsy' else 0)
        print(f"\nLoaded {len(existing_samples)} existing samples from previous sessions")
    except FileNotFoundError:
        print("\nNo previous data found. Starting fresh.")
    
    print("\nInstructions:")
    print("1. The system will detect your drowsiness state")
    print("2. Press keys to label the ACTUAL state:")
    print("   - Press 'a' = Actually AWAKE")
    print("   - Press 'd' = Actually DROWSY")
    print("   - Press 's' = Skip this sample")
    print("3. Press 'q' to quit and update confusion matrix")
    print("4. Press 'r' to RESET all data and start fresh")
    
    if len(existing_samples) > 0:
        print(f"\nCurrent dataset has {len(existing_samples)} samples")
        print("New samples will be added to existing data")
    
    print("\nCalibration will start in 3 seconds...")
    time.sleep(3)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    # Create face landmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # Parameters
    CONSECUTIVE_FRAMES_THRESHOLD = 60
    MAR_YAWN_THRESHOLD = 0.6
    HEAD_PITCH_THRESHOLD = 35.0
    HEAD_TILT_CONSECUTIVE_FRAMES = 90
    calibration_frames = 150
    
    # Variables
    consecutive_low_ear_frames = 0
    consecutive_head_tilt_frames = 0
    smoothing_history = deque(maxlen=5)
    mar_history = deque(maxlen=5)
    baseline_ear_sum = 0.0
    baseline_ear_count = 0
    avg_baseline = None
    
    baseline_pitch_sum = 0.0
    baseline_yaw_sum = 0.0
    baseline_roll_sum = 0.0
    baseline_pose_count = 0
    baseline_pitch = None
    baseline_yaw = None
    baseline_roll = None
    
    frame_count = 0
    
    # Data collection for confusion matrix
    predictions = existing_predictions.copy()
    ground_truth = existing_ground_truth.copy()
    samples_data = existing_samples.copy()
    new_samples_count = 0
    
    print("\nStarting calibration...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_count)
        
        drowsy_status = "Awake"
        avg_ear = 0
        avg_mar = 0
        pitch = 0
        yaw = 0
        roll = 0
        is_yawning = False
        eyes_closed = False
        is_nodding = False
        
        if face_landmarker_result.face_landmarks:
            landmarks = face_landmarker_result.face_landmarks[0]
            
            left_eye = get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
            left_eye = np.array(left_eye)
            left_ear = calculate_ear(left_eye)
            
            right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
            right_eye = np.array(right_eye)
            right_ear = calculate_ear(right_eye)
            
            mouth = get_mouth_landmarks(landmarks, MOUTH_INDICES)
            mouth = np.array(mouth)
            avg_mar = calculate_mar(mouth)
            
            head_pose_points = get_head_pose_landmarks(landmarks, HEAD_POSE_INDICES, w, h)
            pitch, yaw, roll = calculate_head_pose(head_pose_points, w, h)
            
            avg_ear = (left_ear + right_ear) / 2.0
            smoothing_history.append(avg_ear)
            mar_history.append(avg_mar)
            
            smoothed_ear = np.mean(smoothing_history)
            smoothed_mar = np.mean(mar_history)
            
            if frame_count <= calibration_frames:
                baseline_ear_sum += smoothed_ear
                baseline_ear_count += 1
                baseline_pitch_sum += pitch
                baseline_yaw_sum += yaw
                baseline_roll_sum += roll
                baseline_pose_count += 1
            else:
                if avg_baseline is None and baseline_ear_count > 0:
                    avg_baseline = baseline_ear_sum / baseline_ear_count
                
                if baseline_pitch is None and baseline_pose_count > 0:
                    baseline_pitch = baseline_pitch_sum / baseline_pose_count
                    baseline_yaw = baseline_yaw_sum / baseline_pose_count
                    baseline_roll = baseline_roll_sum / baseline_pose_count
                    print(f"\nCalibration complete! Baseline EAR: {avg_baseline:.3f}")
                    print("Start labeling samples now...")
                
                if avg_baseline is not None and smoothed_ear < 0.7 * avg_baseline:
                    eyes_closed = True
                    consecutive_low_ear_frames += 1
                else:
                    eyes_closed = False
                    consecutive_low_ear_frames = 0
                
                if smoothed_mar > MAR_YAWN_THRESHOLD:
                    is_yawning = True
                
                if baseline_pitch is not None:
                    pitch_deviation = pitch - baseline_pitch
                    
                    if pitch_deviation > HEAD_PITCH_THRESHOLD:
                        consecutive_head_tilt_frames += 1
                    else:
                        consecutive_head_tilt_frames = 0
                    
                    if consecutive_head_tilt_frames >= HEAD_TILT_CONSECUTIVE_FRAMES:
                        is_nodding = True
                
                if eyes_closed and is_nodding:
                    drowsy_status = "Drowsy"
                elif eyes_closed and is_yawning:
                    drowsy_status = "Drowsy"
                elif consecutive_low_ear_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                    drowsy_status = "Drowsy"
                elif is_yawning:
                    drowsy_status = "Awake"  # Yawning alone = still awake
                else:
                    drowsy_status = "Awake"
        else:
            drowsy_status = "No face"
        
        # Display
        status_color = (0, 0, 255) if drowsy_status == "Drowsy" else (0, 255, 0)
        
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {avg_mar:.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Predicted: {drowsy_status}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if frame_count > calibration_frames:
            cv2.putText(frame, "Press: 'a'=Awake 'd'=Drowsy 's'=Skip 'r'=Reset", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Total Samples: {len(predictions)} (New: {new_samples_count})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Calibrating... {frame_count}/{calibration_frames}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Model Evaluation - Label Your State", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting and updating confusion matrix...")
            break
        elif key == ord('r'):
            # Reset all data
            response = input("\nAre you sure you want to RESET all data? (yes/no): ")
            if response.lower() == 'yes':
                predictions = []
                ground_truth = []
                samples_data = []
                new_samples_count = 0
                print("All data reset!")
            continue
        elif key == ord('a') and frame_count > calibration_frames:
            # User says they are actually AWAKE
            predictions.append(1 if drowsy_status == "Drowsy" else 0)
            ground_truth.append(0)  # 0 = Awake
            samples_data.append({
                "timestamp": datetime.now().isoformat(),
                "predicted": drowsy_status,
                "actual": "Awake",
                "ear": float(avg_ear),
                "mar": float(avg_mar),
                "pitch": float(pitch)
            })
            new_samples_count += 1
            print(f"Sample {len(predictions)}: Predicted={drowsy_status}, Actual=Awake")
        elif key == ord('d') and frame_count > calibration_frames:
            # User says they are actually DROWSY
            predictions.append(1 if drowsy_status == "Drowsy" else 0)
            ground_truth.append(1)  # 1 = Drowsy
            samples_data.append({
                "timestamp": datetime.now().isoformat(),
                "predicted": drowsy_status,
                "actual": "Drowsy",
                "ear": float(avg_ear),
                "mar": float(avg_mar),
                "pitch": float(pitch)
            })
            new_samples_count += 1
            print(f"Sample {len(predictions)}: Predicted={drowsy_status}, Actual=Drowsy")
        elif key == ord('s'):
            # Skip this sample
            pass
    
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    
    # Generate confusion matrix
    if len(predictions) > 0:
        print(f"\n{new_samples_count} new samples added this session")
        generate_confusion_matrix(predictions, ground_truth, samples_data)
    else:
        print("\nNo samples collected. Please run again and label some samples.")

def generate_confusion_matrix(predictions, ground_truth, samples_data):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX RESULTS")
    print("=" * 60)
    
    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    print(f"\nTotal Samples: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Awake  Drowsy")
    print(f"Actual Awake    {cm[0][0]:3d}    {cm[0][1]:3d}")
    print(f"       Drowsy   {cm[1][0]:3d}    {cm[1][1]:3d}")
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"True Positives (TP):  {tp} - Correctly detected drowsy")
    print(f"True Negatives (TN):  {tn} - Correctly detected awake")
    print(f"False Positives (FP): {fp} - False alarm (predicted drowsy, actually awake)")
    print(f"False Negatives (FN): {fn} - Missed drowsy (predicted awake, actually drowsy)")
    
    print(f"\nPerformance Metrics:")
    print(f"Precision:    {precision:.2%} - When it says drowsy, how often is it right?")
    print(f"Recall:       {recall:.2%} - How many actual drowsy cases did it catch?")
    print(f"F1-Score:     {f1_score:.2%} - Overall balance of precision and recall")
    print(f"Specificity:  {specificity:.2%} - How well does it avoid false alarms?")
    
    # Save data
    with open('evaluation_data.json', 'w') as f:
        json.dump({
            "samples": samples_data,
            "confusion_matrix": cm.tolist(),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "specificity": float(specificity)
            }
        }, f, indent=2)
    print("\nData saved to 'evaluation_data.json'")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Awake', 'Drowsy'],
                yticklabels=['Awake', 'Drowsy'])
    plt.title('Drowsiness Detection Confusion Matrix')
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix plot saved to 'confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    main()
