# Driver Drowsiness Detection

This Python program uses OpenCV and MediaPipe to detect drowsiness in drivers by analyzing multiple indicators: eye closure (EAR), yawning (MAR), and head pose from live webcam feed.

## Features

- Real-time facial landmark detection using MediaPipe FaceMesh
- **Eye Aspect Ratio (EAR)** calculation for eye closure detection
- **Mouth Aspect Ratio (MAR)** calculation for yawning detection
- **Head Pose Estimation** (pitch, yaw, roll) for nodding/head dropping detection
- Multi-modal drowsiness detection with high confidence alerts when 2+ indicators trigger
- Automatic baseline calibration during first 5 seconds
- Live webcam tracking with visual overlay of all metrics and status
- Configurable thresholds for each detection method

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Webcam

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure `face_landmarker.task` model file is in the project directory

## Usage

Run the script:
```bash
python drowsiness_detector.py
```

The program will:
1. Access your webcam
2. Calibrate for 5 seconds (keep your head in normal position with eyes open)
3. Start real-time drowsiness detection
4. Display EAR, MAR, head pose angles, and drowsiness status

Press 'q' to quit.

## Detection Logic

The system monitors three indicators:
- **Eyes Closed**: EAR drops below 70% of calibrated baseline
- **Yawning**: MAR exceeds 0.6 threshold
- **Nodding**: Head pose deviates >15° (pitch) or >20° (yaw) from baseline for 30+ frames

**Status Levels:**
- **Awake** (Green): All indicators normal
- **Yawning** (Orange): Mouth opening detected
- **Nodding** (Orange): Head tilt/drop detected
- **Drowsy** (Red): Eyes closed for 48+ consecutive frames
- **Drowsy (High Confidence)** (Red): 2 or more indicators active simultaneously

## Configuration

Key parameters in `drowsiness_detector.py`:
- `CONSECUTIVE_FRAMES_THRESHOLD`: Frames for eye closure alert (default: 48)
- `MAR_YAWN_THRESHOLD`: Mouth aspect ratio for yawning (default: 0.6)
- `HEAD_PITCH_THRESHOLD`: Pitch deviation in degrees (default: 15.0)
- `HEAD_YAW_THRESHOLD`: Yaw deviation in degrees (default: 20.0)
- `HEAD_TILT_CONSECUTIVE_FRAMES`: Frames before nodding alert (default: 30)
- `calibration_frames`: Calibration duration in frames (default: 150)