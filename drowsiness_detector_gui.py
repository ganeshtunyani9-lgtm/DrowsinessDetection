import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

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

def generate_beep_sound(frequency=440, duration=0.3, sample_rate=22050, volume=0.5):
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    wave = np.sin(frequency * 2 * np.pi * t)
    
    fade_samples = int(sample_rate * 0.01)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    wave[:fade_samples] *= fade_in
    wave[-fade_samples:] *= fade_out
    
    wave = (wave * volume * 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    
    return pygame.sndarray.make_sound(stereo_wave)

def initialize_audio():
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        warning_sound = generate_beep_sound(frequency=800, duration=0.2, volume=0.3)
        alert_sound = generate_beep_sound(frequency=1000, duration=0.5, volume=0.5)
        critical_sound = generate_beep_sound(frequency=1200, duration=0.8, volume=0.7)
        return warning_sound, alert_sound, critical_sound
    except Exception as e:
        print(f"Warning: Could not initialize audio: {e}")
        return None, None, None

class DrowsinessDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Detection state
        self.is_running = False
        self.cap = None
        self.landmarker = None
        
        # Audio
        self.warning_sound, self.alert_sound, self.critical_sound = initialize_audio()
        self.audio_enabled = self.warning_sound is not None
        
        # Create GUI
        self.create_widgets()
        
        # Detection variables
        self.frame_count = 0
        self.calibration_frames = 150
        self.consecutive_low_ear_frames = 0
        self.consecutive_head_tilt_frames = 0
        self.smoothing_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        self.baseline_ear_sum = 0.0
        self.baseline_ear_count = 0
        self.avg_baseline = None
        self.baseline_pitch_sum = 0.0
        self.baseline_yaw_sum = 0.0
        self.baseline_roll_sum = 0.0
        self.baseline_pose_count = 0
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.baseline_roll = None
        self.last_alert_time = 0
        self.alert_active = False
        
        # Statistics
        self.total_alerts = 0
        self.drowsy_detections = 0
        self.yawn_detections = 0
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#34495e', height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🚗 Driver Drowsiness Detection System", 
                               font=('Arial', 20, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=10)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        video_label = tk.Label(left_panel, text="Video Feed", font=('Arial', 12, 'bold'), 
                              bg='#34495e', fg='white')
        video_label.pack(pady=5)
        
        self.video_canvas = tk.Label(left_panel, bg='black')
        self.video_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right panel - Controls and metrics
        right_panel = tk.Frame(main_container, bg='#34495e', width=350, relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        right_panel.pack_propagate(False)
        
        # Control buttons
        control_frame = tk.LabelFrame(right_panel, text="Controls", font=('Arial', 11, 'bold'),
                                      bg='#34495e', fg='white', relief=tk.GROOVE, bd=2)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(control_frame, text="▶ Start Detection", 
                                      command=self.start_detection,
                                      bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                      relief=tk.RAISED, bd=3, cursor='hand2')
        self.start_button.pack(pady=10, padx=10, fill=tk.X)
        
        self.stop_button = tk.Button(control_frame, text="⏹ Stop Detection", 
                                     command=self.stop_detection,
                                     bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                     relief=tk.RAISED, bd=3, cursor='hand2', state=tk.DISABLED)
        self.stop_button.pack(pady=10, padx=10, fill=tk.X)
        
        # Status display
        status_frame = tk.LabelFrame(right_panel, text="Status", font=('Arial', 11, 'bold'),
                                    bg='#34495e', fg='white', relief=tk.GROOVE, bd=2)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(status_frame, text="● Stopped", 
                                     font=('Arial', 16, 'bold'), bg='#34495e', fg='#95a5a6')
        self.status_label.pack(pady=10)
        
        # Metrics display
        metrics_frame = tk.LabelFrame(right_panel, text="Real-time Metrics", 
                                     font=('Arial', 11, 'bold'),
                                     bg='#34495e', fg='white', relief=tk.GROOVE, bd=2)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.ear_label = tk.Label(metrics_frame, text="EAR: --", 
                                 font=('Arial', 11), bg='#34495e', fg='white')
        self.ear_label.pack(anchor=tk.W, padx=10, pady=3)
        
        self.mar_label = tk.Label(metrics_frame, text="MAR: --", 
                                 font=('Arial', 11), bg='#34495e', fg='white')
        self.mar_label.pack(anchor=tk.W, padx=10, pady=3)
        
        self.pitch_label = tk.Label(metrics_frame, text="Pitch: --", 
                                   font=('Arial', 11), bg='#34495e', fg='white')
        self.pitch_label.pack(anchor=tk.W, padx=10, pady=3)
        
        self.yaw_label = tk.Label(metrics_frame, text="Yaw: --", 
                                 font=('Arial', 11), bg='#34495e', fg='white')
        self.yaw_label.pack(anchor=tk.W, padx=10, pady=3)
        
        # Statistics
        stats_frame = tk.LabelFrame(right_panel, text="Session Statistics", 
                                   font=('Arial', 11, 'bold'),
                                   bg='#34495e', fg='white', relief=tk.GROOVE, bd=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.alerts_label = tk.Label(stats_frame, text="Total Alerts: 0", 
                                    font=('Arial', 10), bg='#34495e', fg='white')
        self.alerts_label.pack(anchor=tk.W, padx=10, pady=3)
        
        self.drowsy_label = tk.Label(stats_frame, text="Drowsy Events: 0", 
                                    font=('Arial', 10), bg='#34495e', fg='white')
        self.drowsy_label.pack(anchor=tk.W, padx=10, pady=3)
        
        self.yawn_label = tk.Label(stats_frame, text="Yawn Events: 0", 
                                  font=('Arial', 10), bg='#34495e', fg='white')
        self.yawn_label.pack(anchor=tk.W, padx=10, pady=3)
        
        # Audio status
        audio_status = "🔊 Audio: Enabled" if self.audio_enabled else "🔇 Audio: Disabled"
        audio_label = tk.Label(right_panel, text=audio_status, 
                              font=('Arial', 9), bg='#34495e', fg='#95a5a6')
        audio_label.pack(side=tk.BOTTOM, pady=10)
        
    def start_detection(self):
        if self.is_running:
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot access webcam!")
                return
            
            # Initialize MediaPipe
            base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            
            # Reset variables
            self.frame_count = 0
            self.consecutive_low_ear_frames = 0
            self.consecutive_head_tilt_frames = 0
            self.smoothing_history.clear()
            self.mar_history.clear()
            self.baseline_ear_sum = 0.0
            self.baseline_ear_count = 0
            self.avg_baseline = None
            self.baseline_pitch_sum = 0.0
            self.baseline_yaw_sum = 0.0
            self.baseline_roll_sum = 0.0
            self.baseline_pose_count = 0
            self.baseline_pitch = None
            self.baseline_yaw = None
            self.baseline_roll = None
            self.last_alert_time = 0
            self.alert_active = False
            self.total_alerts = 0
            self.drowsy_detections = 0
            self.yawn_detections = 0
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="● Calibrating...", fg='#f39c12')
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {e}")
            self.stop_detection()
    
    def stop_detection(self):
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="● Stopped", fg='#95a5a6')
        self.video_canvas.config(image='')
        
    def detection_loop(self):
        CONSECUTIVE_FRAMES_THRESHOLD = 60
        MAR_YAWN_THRESHOLD = 0.6
        HEAD_PITCH_THRESHOLD = 35.0
        HEAD_TILT_CONSECUTIVE_FRAMES = 90
        ALERT_COOLDOWN = 3.0
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            face_landmarker_result = self.landmarker.detect_for_video(mp_image, self.frame_count)
            
            drowsy_status = "Awake"
            avg_ear = 0
            avg_mar = 0
            pitch = 0
            yaw = 0
            is_yawning = False
            eyes_closed = False
            is_nodding = False
            
            if face_landmarker_result.face_landmarks:
                landmarks = face_landmarker_result.face_landmarks[0]
                
                left_eye = np.array(get_eye_landmarks(landmarks, LEFT_EYE_INDICES))
                right_eye = np.array(get_eye_landmarks(landmarks, RIGHT_EYE_INDICES))
                mouth = np.array(get_mouth_landmarks(landmarks, MOUTH_INDICES))
                
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                avg_mar = calculate_mar(mouth)
                
                head_pose_points = get_head_pose_landmarks(landmarks, HEAD_POSE_INDICES, w, h)
                pitch, yaw, roll = calculate_head_pose(head_pose_points, w, h)
                
                self.smoothing_history.append(avg_ear)
                self.mar_history.append(avg_mar)
                
                smoothed_ear = np.mean(self.smoothing_history)
                smoothed_mar = np.mean(self.mar_history)
                
                if self.frame_count <= self.calibration_frames:
                    self.baseline_ear_sum += smoothed_ear
                    self.baseline_ear_count += 1
                    self.baseline_pitch_sum += pitch
                    self.baseline_yaw_sum += yaw
                    self.baseline_roll_sum += roll
                    self.baseline_pose_count += 1
                else:
                    if self.avg_baseline is None and self.baseline_ear_count > 0:
                        self.avg_baseline = self.baseline_ear_sum / self.baseline_ear_count
                        self.baseline_pitch = self.baseline_pitch_sum / self.baseline_pose_count
                        self.baseline_yaw = self.baseline_yaw_sum / self.baseline_pose_count
                        self.baseline_roll = self.baseline_roll_sum / self.baseline_pose_count
                        self.root.after(0, lambda: self.status_label.config(text="● Active", fg='#27ae60'))
                    
                    if self.avg_baseline is not None and smoothed_ear < 0.7 * self.avg_baseline:
                        eyes_closed = True
                        self.consecutive_low_ear_frames += 1
                    else:
                        eyes_closed = False
                        self.consecutive_low_ear_frames = 0
                    
                    if smoothed_mar > MAR_YAWN_THRESHOLD:
                        is_yawning = True
                    
                    if self.baseline_pitch is not None:
                        pitch_deviation = pitch - self.baseline_pitch
                        if pitch_deviation > HEAD_PITCH_THRESHOLD:
                            self.consecutive_head_tilt_frames += 1
                        else:
                            self.consecutive_head_tilt_frames = 0
                        
                        if self.consecutive_head_tilt_frames >= HEAD_TILT_CONSECUTIVE_FRAMES:
                            is_nodding = True
                    
                    if eyes_closed and is_nodding:
                        drowsy_status = "Drowsy (High Confidence)"
                    elif eyes_closed and is_yawning:
                        drowsy_status = "Drowsy (High Confidence)"
                    elif self.consecutive_low_ear_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                        drowsy_status = "Drowsy"
                    elif is_yawning:
                        drowsy_status = "Yawning"
                    else:
                        drowsy_status = "Awake"
                    
                    # Audio alerts
                    current_time = time.time()
                    if self.audio_enabled and (current_time - self.last_alert_time) >= ALERT_COOLDOWN:
                        if drowsy_status == "Drowsy (High Confidence)":
                            self.critical_sound.play()
                            self.last_alert_time = current_time
                            self.alert_active = True
                            self.total_alerts += 1
                            self.drowsy_detections += 1
                        elif drowsy_status == "Drowsy":
                            self.alert_sound.play()
                            self.last_alert_time = current_time
                            self.alert_active = True
                            self.total_alerts += 1
                            self.drowsy_detections += 1
                        elif drowsy_status == "Yawning":
                            self.warning_sound.play()
                            self.last_alert_time = current_time
                            self.alert_active = True
                            self.total_alerts += 1
                            self.yawn_detections += 1
                        else:
                            self.alert_active = False
                    else:
                        if (current_time - self.last_alert_time) >= ALERT_COOLDOWN:
                            self.alert_active = False
            
            # Update GUI
            self.root.after(0, self.update_metrics, avg_ear, avg_mar, pitch, yaw, drowsy_status)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda: self.video_canvas.config(image=imgtk))
            self.video_canvas.imgtk = imgtk
            
            time.sleep(0.03)
    
    def update_metrics(self, ear, mar, pitch, yaw, status):
        self.ear_label.config(text=f"EAR: {ear:.3f}")
        self.mar_label.config(text=f"MAR: {mar:.3f}")
        self.pitch_label.config(text=f"Pitch: {pitch:.1f}°")
        self.yaw_label.config(text=f"Yaw: {yaw:.1f}°")
        
        if "Drowsy" in status:
            self.status_label.config(text=f"● {status}", fg='#e74c3c')
        elif status == "Yawning":
            self.status_label.config(text=f"● {status}", fg='#f39c12')
        else:
            self.status_label.config(text=f"● {status}", fg='#27ae60')
        
        self.alerts_label.config(text=f"Total Alerts: {self.total_alerts}")
        self.drowsy_label.config(text=f"Drowsy Events: {self.drowsy_detections}")
        self.yawn_label.config(text=f"Yawn Events: {self.yawn_detections}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectorGUI(root)
    root.mainloop()
    
    if app.audio_enabled:
        pygame.mixer.quit()
