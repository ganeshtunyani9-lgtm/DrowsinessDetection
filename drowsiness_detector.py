import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import time

# Eye landmarks indices for FaceMesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 362, 380, 374]

# Mouth landmarks indices for FaceMesh
# Outer lips: top, bottom, left, right
MOUTH_INDICES = [61, 291, 78, 308, 13, 14]  # top, bottom, left, right points

# Head pose landmarks indices
# Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
HEAD_POSE_INDICES = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose estimation (generic face model)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for an eye.
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    where p1-p6 are eye landmark points
    """
    if len(eye_landmarks) < 6:
        return 0
    
    # Vertical distances
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR calculation
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    return ear

def calculate_mar(mouth_landmarks):
    """
    Calculate Mouth Aspect Ratio (MAR) for yawning detection.
    
    MAR = vertical lip distance / horizontal mouth distance
    Higher MAR indicates mouth opening (yawning)
    """
    if len(mouth_landmarks) < 6:
        return 0
    
    # Vertical distance (top to bottom lip)
    vertical_1 = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    vertical_2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[5])
    
    # Horizontal distance (left to right corner)
    horizontal = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
    
    # MAR calculation
    mar = (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0
    return mar

def get_eye_landmarks(landmarks, indices):
    """Extract specific eye landmarks from all face landmarks."""
    eye_points = []
    for idx in indices:
        point = landmarks[idx]
        eye_points.append(np.array([point.x, point.y]))
    return eye_points

def get_mouth_landmarks(landmarks, indices):
    """Extract specific mouth landmarks from all face landmarks."""
    mouth_points = []
    for idx in indices:
        point = landmarks[idx]
        mouth_points.append(np.array([point.x, point.y]))
    return mouth_points

def get_head_pose_landmarks(landmarks, indices, width, height):
    """Extract head pose landmarks and convert to pixel coordinates."""
    image_points = []
    for idx in indices:
        point = landmarks[idx]
        x = point.x * width
        y = point.y * height
        image_points.append([x, y])
    return np.array(image_points, dtype=np.float64)

def calculate_head_pose(image_points, width, height):
    """
    Calculate head pose angles (pitch, yaw, roll) using solvePnP.
    
    Returns:
        pitch: Head tilt up/down (nodding)
        yaw: Head turn left/right
        roll: Head tilt side to side
    """
    # Camera internals (approximate)
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0, 0, 0
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles from rotation matrix
    # Extract pitch, yaw, roll
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
    
    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    
    return pitch, yaw, roll

def generate_beep_sound(frequency=440, duration=0.3, sample_rate=22050, volume=0.5):
    """Generate a beep sound using sine wave."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    wave = np.sin(frequency * 2 * np.pi * t)
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(sample_rate * 0.01)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    wave[:fade_samples] *= fade_in
    wave[-fade_samples:] *= fade_out
    
    # Convert to 16-bit audio
    wave = (wave * volume * 32767).astype(np.int16)
    
    # Stereo sound
    stereo_wave = np.column_stack((wave, wave))
    
    return pygame.sndarray.make_sound(stereo_wave)

def initialize_audio():
    """Initialize pygame mixer and create alert sounds."""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Generate different alert sounds
        warning_sound = generate_beep_sound(frequency=800, duration=0.2, volume=0.3)  # Soft beep
        alert_sound = generate_beep_sound(frequency=1000, duration=0.5, volume=0.5)  # Loud beep
        critical_sound = generate_beep_sound(frequency=1200, duration=0.8, volume=0.7)  # Critical alarm
        
        return warning_sound, alert_sound, critical_sound
    except Exception as e:
        print(f"Warning: Could not initialize audio: {e}")
        return None, None, None

def main():
    # Initialize audio system
    warning_sound, alert_sound, critical_sound = initialize_audio()
    audio_enabled = warning_sound is not None
    
    if audio_enabled:
        print("Audio alerts enabled")
    else:
        print("Audio alerts disabled (pygame mixer not available)")
    
    # Use webcam for live tracking
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    # Set calibration duration (assuming ~30 fps, 5 seconds = 150 frames)
    calibration_frames = 150  # 5 seconds - quick calibration for drivers
    print(f"Calibration will use first {calibration_frames} frames (~5 seconds)")
    print("Look straight ahead at the road in your normal driving position")
    print("Keep eyes open and head in comfortable position")
    
    # Create face landmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # Drowsiness detection parameters
    CONSECUTIVE_FRAMES_THRESHOLD = 60  # number of frames for eye closure alert (2 seconds at 30fps)
    MAR_YAWN_THRESHOLD = 0.6  # threshold for yawning detection
    HEAD_PITCH_THRESHOLD = 35.0  # degrees for head pitch deviation (forward nodding/dropping)
    HEAD_YAW_THRESHOLD = 50.0  # degrees for head yaw deviation (turning) - very high for normal driving movements
    HEAD_TILT_CONSECUTIVE_FRAMES = 90  # frames before marking as nodding - 3 seconds for sustained drop
    ALERT_COOLDOWN = 3.0  # seconds between alerts to prevent spam
    
    # calibration variables
    consecutive_low_ear_frames = 0
    consecutive_head_tilt_frames = 0
    ear_history = deque(maxlen=30)       # overall history
    smoothing_history = deque(maxlen=5)  # for moving average
    mar_history = deque(maxlen=5)        # for MAR smoothing
    baseline_ear_sum = 0.0
    baseline_ear_count = 0
    avg_baseline = None
    
    # Head pose baseline calibration
    baseline_pitch_sum = 0.0
    baseline_yaw_sum = 0.0
    baseline_roll_sum = 0.0
    baseline_pose_count = 0
    baseline_pitch = None
    baseline_yaw = None
    baseline_roll = None
    
    # Alert system variables
    last_alert_time = 0
    alert_active = False
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        frame_count += 1
        
        # Flip frame for selfie view (optional)
        # frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions for landmark scaling
        h, w, c = frame.shape
        
        # Detect face landmarks
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
            
            # Get left eye landmarks
            left_eye = get_eye_landmarks(landmarks, LEFT_EYE_INDICES)
            left_eye = np.array(left_eye)
            left_ear = calculate_ear(left_eye)
            
            # Get right eye landmarks
            right_eye = get_eye_landmarks(landmarks, RIGHT_EYE_INDICES)
            right_eye = np.array(right_eye)
            right_ear = calculate_ear(right_eye)
            
            # Get mouth landmarks
            mouth = get_mouth_landmarks(landmarks, MOUTH_INDICES)
            mouth = np.array(mouth)
            avg_mar = calculate_mar(mouth)
            
            # Get head pose
            head_pose_points = get_head_pose_landmarks(landmarks, HEAD_POSE_INDICES, w, h)
            pitch, yaw, roll = calculate_head_pose(head_pose_points, w, h)
            
            # Average EAR (raw)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_history.append(avg_ear)
            smoothing_history.append(avg_ear)
            mar_history.append(avg_mar)
            
            # compute smoothed EAR and MAR using last 5 frames (or fewer if not available)
            smoothed_ear = np.mean(smoothing_history)
            smoothed_mar = np.mean(mar_history)
            
            # calibration phase (collect baseline silently)
            if frame_count <= calibration_frames:
                baseline_ear_sum += smoothed_ear
                baseline_ear_count += 1
                # Collect head pose baseline
                baseline_pitch_sum += pitch
                baseline_yaw_sum += yaw
                baseline_roll_sum += roll
                baseline_pose_count += 1
            else:
                # Calculate baselines after calibration
                if avg_baseline is None and baseline_ear_count > 0:
                    avg_baseline = baseline_ear_sum / baseline_ear_count
                
                if baseline_pitch is None and baseline_pose_count > 0:
                    baseline_pitch = baseline_pitch_sum / baseline_pose_count
                    baseline_yaw = baseline_yaw_sum / baseline_pose_count
                    baseline_roll = baseline_roll_sum / baseline_pose_count
                    print(f"Head pose baseline calibrated: Pitch={baseline_pitch:.1f}, Yaw={baseline_yaw:.1f}, Roll={baseline_roll:.1f}")
                
                # Detect eyes closed
                if avg_baseline is not None and smoothed_ear < 0.7 * avg_baseline:
                    eyes_closed = True
                    consecutive_low_ear_frames += 1
                else:
                    eyes_closed = False
                    consecutive_low_ear_frames = 0
                
                # Detect yawning
                if smoothed_mar > MAR_YAWN_THRESHOLD:
                    is_yawning = True
                
                # Detect head nodding/dropping (compare against baseline)
                if baseline_pitch is not None:
                    pitch_deviation = pitch - baseline_pitch  # Positive = head tilted forward/down
                    yaw_deviation = abs(yaw - baseline_yaw)
                    
                    # Only detect forward pitch (nodding/dropping), ignore yaw (normal driving)
                    # Drowsy nodding is sustained forward head drop, not quick movements
                    if pitch_deviation > HEAD_PITCH_THRESHOLD:  # Head dropped forward significantly
                        consecutive_head_tilt_frames += 1
                    else:
                        consecutive_head_tilt_frames = 0
                    
                    # Only mark as nodding if sustained forward drop
                    if consecutive_head_tilt_frames >= HEAD_TILT_CONSECUTIVE_FRAMES:
                        is_nodding = True
                
                # Combined drowsiness detection logic
                # Count how many drowsiness indicators are active
                drowsy_indicators = sum([eyes_closed, is_yawning, is_nodding])
                
                # High priority: Eyes closed + nodding = critical drowsiness (micro-sleep)
                if eyes_closed and is_nodding:
                    drowsy_status = "Drowsy (High Confidence)"
                # Eyes closed + yawning = high drowsiness
                elif eyes_closed and is_yawning:
                    drowsy_status = "Drowsy (High Confidence)"
                # Just eyes closed for extended time
                elif consecutive_low_ear_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                    drowsy_status = "Drowsy"
                # Just yawning (fatigue warning)
                elif is_yawning:
                    drowsy_status = "Yawning"
                # Nodding alone doesn't trigger alert (could be normal movement)
                # Only alerts when combined with eyes closed
                else:
                    drowsy_status = "Awake"
                
                # Audio alert system
                current_time = time.time()
                if audio_enabled and (current_time - last_alert_time) >= ALERT_COOLDOWN:
                    if drowsy_status == "Drowsy (High Confidence)":
                        critical_sound.play()
                        last_alert_time = current_time
                        alert_active = True
                    elif drowsy_status == "Drowsy":
                        alert_sound.play()
                        last_alert_time = current_time
                        alert_active = True
                    elif drowsy_status == "Yawning":
                        warning_sound.play()
                        last_alert_time = current_time
                        alert_active = True
                    else:
                        alert_active = False
                else:
                    # Check if cooldown expired
                    if (current_time - last_alert_time) >= ALERT_COOLDOWN:
                        alert_active = False
            
            # Draw eye landmarks on frame
            for point in left_eye:
                x = int(point[0] * w)
                y = int(point[1] * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            for point in right_eye:
                x = int(point[0] * w)
                y = int(point[1] * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw mouth landmarks on frame
            for point in mouth:
                x = int(point[0] * w)
                y = int(point[1] * h)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        else:
            # no face detected this frame
            drowsy_status = "No face"        
        # Determine color based on status
        if "Drowsy" in drowsy_status:
            status_color = (0, 0, 255)  # Red
        elif drowsy_status == "Yawning":
            status_color = (0, 165, 255)  # Orange
        else:
            status_color = (0, 255, 0)  # Green
        
        # Display EAR value
        ear_text = f"EAR: {avg_ear:.3f}"
        cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        
        # Display MAR value
        mar_text = f"MAR: {avg_mar:.3f}"
        cv2.putText(frame, mar_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        
        # Display head pose angles (with deviation from baseline if available)
        if baseline_pitch is not None:
            pitch_dev = pitch - baseline_pitch
            yaw_dev = yaw - baseline_yaw
            pose_text = f"Pitch: {pitch:.1f} ({pitch_dev:+.1f}) Yaw: {yaw:.1f} ({yaw_dev:+.1f})"
        else:
            pose_text = f"Pitch: {pitch:.1f} Yaw: {yaw:.1f} Roll: {roll:.1f}"
        cv2.putText(frame, pose_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        
        # Display drowsiness status
        status_text = f"Status: {drowsy_status}"
        cv2.putText(frame, status_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    1, status_color, 2)
        
        # Display frame count
        frame_text = f"Frame: {frame_count}"
        cv2.putText(frame, frame_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        
        # Display alert indicator
        if alert_active and audio_enabled:
            alert_indicator = "ALERT ACTIVE"
            cv2.putText(frame, alert_indicator, (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        # Exit on 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    
    if audio_enabled:
        pygame.mixer.quit()

if __name__ == "__main__":
    main()