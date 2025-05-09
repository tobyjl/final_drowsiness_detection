import os
import sys
import threading
import time

import cv2
import dlib
import joblib
import numpy as np
from loguru import logger
from playsound import playsound
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import distance

from drowsiness_detection import predict_drowsiness_cnn
from eye_detection import EyeStateDetector, FaceEyeDetector


def calculate_eye_aspect_ratio(eye_points):
    dist_a = distance.euclidean(eye_points[1], eye_points[5])
    dist_b = distance.euclidean(eye_points[2], eye_points[4])
    horiz = distance.euclidean(eye_points[0], eye_points[3])
    return (dist_a + dist_b) / (2.0 * horiz) if horiz != 0 else 0


def calculate_mouth_aspect_ratio(mouth_points):
    dist_a = distance.euclidean(mouth_points[2], mouth_points[10])
    dist_b = distance.euclidean(mouth_points[3], mouth_points[9])
    dist_c = distance.euclidean(mouth_points[4], mouth_points[8])
    horiz = distance.euclidean(mouth_points[0], mouth_points[6])
    return (dist_a + dist_b + dist_c) / (3.0 * horiz) if horiz != 0 else 0


def detect_landmarks_frame(
    image, predictor_path="shape_predictor_68_face_landmarks.dat"
):
    if image is None:
        print("Error: input image is None")
        return None
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return None
    faces = detector(grey, 1)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(grey, face)
    points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)])
    feat = {}
    left_eye_idx = list(range(36, 42))
    right_eye_idx = list(range(42, 48))
    mouth_idx = list(range(48, 68))
    left_ear = calculate_eye_aspect_ratio(points[left_eye_idx])
    right_ear = calculate_eye_aspect_ratio(points[right_eye_idx])
    avg_ear = (left_ear + right_ear) / 2.0
    feat["left_ear"] = left_ear
    feat["right_ear"] = right_ear
    feat["avg_ear"] = avg_ear
    feat["ear_diff"] = abs(left_ear - right_ear)
    mar = calculate_mouth_aspect_ratio(points[mouth_idx])
    feat["mar"] = mar
    face_width = distance.euclidean(points[0], points[16])
    nose_tip = points[30]
    chin = points[8]
    pitch = distance.euclidean(nose_tip, chin) / face_width if face_width != 0 else 0
    feat["head_pitch"] = pitch
    left_ear_pt = points[0]
    right_ear_pt = points[16]
    nose_bridge = points[27]
    left_dist = distance.euclidean(left_ear_pt, nose_bridge)
    right_dist = distance.euclidean(right_ear_pt, nose_bridge)
    feat["head_yaw"] = (
        abs(left_dist - right_dist) / face_width if face_width != 0 else 0
    )
    feat["emr"] = avg_ear / (mar + 0.01)
    return feat


def get_feature_vector(features_dict, feature_names):
    return np.array([features_dict.get(feat, 0) for feat in feature_names])


class AudioManager:
    def __init__(self, logger):
        self.logger = logger
        self.audio_folder = "inputs/audio"
        self.audio_map = {
            "Monitoring started": "monitoring_started.mp3",
            "Monitoring stopped": "monitoring_stopped.mp3",
            "Running pre-drive drowsiness assessment": "assessment_pre_drive.mp3",
            "No face detected during assessment": "no_face_detected.mp3",
            "Warning! Drowsiness detected. Not safe to drive.": "assesment_complete_not_safe.mp3",
            "Assessment complete. You appear alert. Safe to drive.": "assessment_complete_safe.mp3",
            "Camera connected": "camera_connected.mp3",
            "Camera connection failed": "camera_failed.mp3",
            "MICROSLEEP! MICROSLEEP! MICROSLEEP!": "microsleep_microsleep_microsleep.mp3",
            "No face detected": "no_face_detected.mp3",
            "Running pre-drive check": "assessment_pre_drive.mp3",
        }

    def play_audio(self, message_key):
        mp3_file = self.audio_map.get(message_key, None)
        if not mp3_file:
            self.logger.warning(
                f"No pre-recorded audio mapped for message: '{message_key}'"
            )
            return
        audio_path = os.path.join(self.audio_folder, mp3_file)
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found at: {audio_path}")
            return
        try:
            self.logger.debug(
                f"Playing audio for message: {message_key} from {audio_path}"
            )
            playsound(audio_path)
        except Exception as e:
            self.logger.error(f"Error playing audio file {audio_path}: {e}")


class CircleWidget(QWidget):
    def __init__(self, color="green", parent=None):
        super().__init__(parent)
        self.colour = QColor(color)
        self.setMinimumSize(30, 30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.colour)
        painter.setPen(Qt.NoPen)
        center = self.rect().center()
        radius = min(self.width(), self.height()) / 2.5
        painter.drawEllipse(center, radius, radius)

    def setColor(self, color):
        if isinstance(color, str):
            if color.lower() == "red":
                self.colour = QColor(231, 76, 60)
            elif color.lower() == "green":
                self.colour = QColor(46, 204, 113)
            else:
                self.colour = QColor(color)
        else:
            self.colour = color
        self.update()


class DrowsinessDetectionDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Monitoring System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2c3e50; color: white;")
        self.logger = logger
        self.audio_manager = AudioManager(self.logger)
        self.microsleep_active = False
        self.is_monitoring = False
        self.camera_index = 2
        self.cap = None
        self.simulation_running = True
        self.predictor_path = "inputs/utils/shape_predictor_68_face_landmarks.dat"
        self.load_ml_models()
        self._tts_lock = threading.Lock()
        self.eyes_indicator_on = True
        self.initialise_ui()
        self.start_processing()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)
        self.eyes_closed_frame_cnt = 0
        self.required_closed_frames = 10
        self.eyes_indicator_blinking = False
        self.eyes_blink_timer = QTimer(self)
        self.eyes_blink_timer.setInterval(500)
        self.eyes_blink_timer.timeout.connect(self.toggle_eyes_indicator)
        self.face_eye_detector = FaceEyeDetector()
        self.eye_state_detector = EyeStateDetector(
            model_path="outputs/models/resnet/best_eyes_model.pth"
        )
        self.eye_state_history = []
        self.max_history = 10
        self.eyes_closed_threshold = 0.6
        self.last_eye_state = None
        self.blink_possible = False
        self.blink_timer_start = None
        self.blink_count = 0
        self.blink_window_start = time.time()
        self.pred_interval = 30
        self.next_pred = time.time() + 30
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._update_countdown)
        self.countdown_timer.start(1000)
        self.last_frame = None

    def load_ml_models(self):
        try:
            self.xgb_model = joblib.load(
                "outputs/models/xgboost/focused_xgb_model.joblib"
            )
            self.scaler = joblib.load("outputs/models/xgboost/focused_scaler.joblib")
            with open("outputs/models/xgboost/focused_threshold.txt", "r") as f:
                self.threshold = float(f.read().strip())
            with open("outputs/models/xgboost/focused_features.txt", "r") as f:
                self.feature_names = f.read().splitlines()
            self.logger.info("ML models loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}")
            self.xgb_model = None
            self.scaler = None
            self.threshold = 0.5
            self.feature_names = []

    def initialise_ui(self):
        self.logger.info("Initialising UI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #34495e;")
        header_layout = QHBoxLayout(header_frame)
        title = QLabel("Driver Drowsiness Monitoring System")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title, stretch=1)
        eyes_label = QLabel("Eyes Status")
        eyes_label.setStyleSheet("font-size: 12px; color: white;")
        self.eyes_indicator = CircleWidget("green")
        self.eyes_indicator.setFixedSize(30, 30)
        header_layout.addWidget(eyes_label)
        header_layout.addWidget(self.eyes_indicator)
        main_layout.addWidget(header_frame)
        indicators_frame = QFrame()
        indicators_layout = QGridLayout(indicators_frame)
        self.indicators = {}
        indicator_names = [
            "Microsleep Detection",
            "XGBoost Assessment",
            "Monitoring ML",
        ]
        for i, name in enumerate(indicator_names):
            indicator_layout = QVBoxLayout()
            circle = CircleWidget("green")
            circle.setFixedSize(30, 30)
            label = QLabel(name)
            label.setStyleSheet("font-size: 10px;")
            label.setAlignment(Qt.AlignCenter)
            indicator_layout.addWidget(circle, alignment=Qt.AlignCenter)
            indicator_layout.addWidget(label, alignment=Qt.AlignCenter)
            indicators_layout.addLayout(indicator_layout, 0, i)
            self.indicators[name] = circle
        main_layout.addWidget(indicators_frame)
        content_frame = QFrame()
        content_layout = QHBoxLayout(content_frame)
        video_frame = QFrame()
        video_frame.setStyleSheet(
            "background-color: #34495e; min-width: 400px; min-height: 300px;"
        )
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        video_layout.addWidget(self.video_label)
        content_layout.addWidget(video_frame)
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background-color: #34495e;")
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_title = QLabel("Current Metrics")
        metrics_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        metrics_title.setAlignment(Qt.AlignCenter)
        metrics_layout.addWidget(metrics_title)
        self.metric_values = {}
        metrics = [
            ("Eye Aspect Ratio (EAR):", "TBD", "Simulation pending"),
            ("Mouth Aspect Ratio (MAR):", "TBD", "Simulation pending"),
            ("Blink Rate (per min):", "Start Monitoring", "Simulation pending"),
            ("Drowsiness Probability:", "TBD", "Simulation pending"),
        ]
        for label_text, initial_value, note in metrics:
            metric_layout = QHBoxLayout()
            name_label = QLabel(label_text)
            name_label.setStyleSheet("font-size: 11px;")
            name_label.setFixedWidth(180)
            value_label = QLabel(initial_value)
            value_label.setStyleSheet("font-size: 11px; font-weight: bold;")
            value_label.setFixedWidth(60)
            range_label = QLabel(f"Note: {note}")
            range_label.setStyleSheet("font-size: 10px; color: #95a5a6;")
            metric_layout.addWidget(name_label)
            metric_layout.addWidget(value_label)
            metric_layout.addWidget(range_label)
            metrics_layout.addLayout(metric_layout)
            self.metric_values[label_text] = value_label
        content_layout.addWidget(metrics_frame)
        main_layout.addWidget(content_frame, 1)
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        self.status_message = QLabel("System Ready")
        self.status_message.setStyleSheet("font-size: 12px; color: #2ecc71;")
        self.status_message.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.status_message)
        self.countdown_label = QLabel("Next update in: 30s")
        self.countdown_label.setStyleSheet("font-size:12px;color:#f1c40f;")
        controls_layout.addWidget(self.countdown_label)
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.setStyleSheet(
            "background-color: #2ecc71; color: black; font-size: 12px; padding: 5px 10px; border: none; border-radius: 3px;"
        )
        self.start_btn.clicked.connect(self.toggle_monitoring)
        pre_drive_btn = QPushButton("Run Pre-Drive Check")
        pre_drive_btn.setStyleSheet(
            "background-color: #3498db; color: black; font-size: 12px; padding: 5px 10px; border: none; border-radius: 3px;"
        )
        pre_drive_btn.clicked.connect(self.run_predrive_check)
        alert_btn = QPushButton("Simulate Alert")
        alert_btn.setStyleSheet(
            "background-color: #e74c3c; color: black; font-size: 12px; padding: 5px 10px; border: none; border-radius: 3px;"
        )
        alert_btn.clicked.connect(self.simulate_microsleep)
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(pre_drive_btn)
        buttons_layout.addWidget(alert_btn)
        controls_layout.addLayout(buttons_layout)
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        camera_label.setStyleSheet("font-size: 11px;")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2", "3"])
        self.camera_combo.setCurrentText(str(self.camera_index))
        self.camera_combo.setFixedWidth(50)
        change_camera_btn = QPushButton("Change Camera")
        change_camera_btn.setStyleSheet(
            "background-color: #7f8c8d; color: black; font-size: 10px; padding: 5px; border: none; border-radius: 3px;"
        )
        change_camera_btn.clicked.connect(self.change_camera)
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(change_camera_btn)
        camera_layout.addStretch(1)
        controls_layout.addLayout(camera_layout)
        main_layout.addWidget(controls_frame)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #e74c3c; font-size: 10px;")
        self.error_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.error_label)
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._update_countdown)
        self.countdown_timer.start(1000)
        self.last_frame = None
        self.logger.info("UI initialisation complete")

    def start_processing(self):
        self.logger.info("Starting processing")
        try:
            self.setup_camera()
            self.processing_thread = threading.Thread(target=self.processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.info("Processing thread started")
        except Exception as e:
            self.logger.error(f"Error starting processing: {e}")
            self.show_error_message(f"Failed to start processing: {e}")

    def setup_camera(self):
        self.logger.info(
            f"Setting up camera with index {self.camera_combo.currentText()}"
        )
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.info("Released previous camera")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
        try:
            self.camera_index = int(self.camera_combo.currentText())
            self.logger.info(f"Opening camera at index {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera at index {self.camera_index}")
                self.status_message.setText(
                    f"Camera error! Check index {self.camera_index}"
                )
                self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.logger.info(f"Camera {self.camera_index} connected")
            self.status_message.setText(f"Camera {self.camera_index} connected")
            self.status_message.setStyleSheet("color: #2ecc71; font-size: 12px;")
            self.play_pre_recorded("Camera connected")
            return True
        except Exception as e:
            self.logger.error(f"Camera error: {str(e)}")
            self.status_message.setText("Camera error! See logs")
            self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
            self.play_pre_recorded("Camera connection failed")
            return False

    def change_camera(self):
        self.logger.info(f"Changing camera to index {self.camera_combo.currentText()}")
        if self.setup_camera():
            self.logger.info(f"Changed to camera {self.camera_index}")
            self.play_pre_recorded("Camera connected")
        else:
            self.logger.error("Failed to change camera")
            self.play_pre_recorded("Camera connection failed")

    def run_predrive_check(self):
        self.logger.info("Running pre drive check")
        self.status_message.setText("Running Pre-Drive Assessment...")
        self.status_message.setStyleSheet("color: #f39c12; font-size: 12px;")
        QApplication.processEvents()
        self.play_pre_recorded("Running pre drive check")
        if self.cap is None or not self.cap.isOpened():
            self.logger.warning("Camera not available for assessment")
            if not self.setup_camera():
                self.status_message.setText("Camera not available for assessment")
                self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
                self.play_pre_recorded("Camera connection failed")
                return
        frame_captured = False
        for _ in range(10):
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame_captured = True
                    self.update_frame_display(frame)
                    QApplication.processEvents()
                    time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error capturing initial frame: {e}")
        if not frame_captured:
            self.logger.error("Failed to capture any frames")
            self.status_message.setText("Failed to capture image for assessment")
            self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read assessment frame")
                self.status_message.setText("Failed to capture image for assessment")
                self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
                return
            self.update_frame_display(frame)
            QApplication.processEvents()
        except Exception as e:
            self.logger.error(f"Error capturing assessment frame: {e}")
            self.status_message.setText("Error during frame capture")
            self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
            return
        features = detect_landmarks_frame(frame, predictor_path=self.predictor_path)
        if features is None:
            self.logger.error("No face detected in pre-drive check")
            self.status_message.setText("No face detected!")
            self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
            self.play_pre_recorded("No face detected during assessment")
            return
        feature_vector = get_feature_vector(features, self.feature_names).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        prob_drowsy = self.xgb_model.predict_proba(feature_vector_scaled)[0, 1]
        self.logger.info(f"Pre-drive drowsiness probability: {prob_drowsy:.4f}")
        if prob_drowsy >= self.threshold:
            result = "Pre-Drive Assessment: Drowsiness Detected - Not Safe to Drive"
            colour = "#e74c3c"
            self.update_indicator("XGBoost Assessment", "red")
            self.play_pre_recorded("Warning! Drowsiness detected. Not safe to drive.")
        else:
            result = "Pre-Drive Assessment: Safe to Drive"
            colour = "#2ecc71"
            self.update_indicator("XGBoost Assessment", "green")
            self.play_pre_recorded(
                "Assessment complete. You appear alert. Safe to drive."
            )
        self.status_message.setText(result)
        self.status_message.setStyleSheet(f"color: {colour}; font-size: 12px;")
        self.logger.info("Pre-drive check completed")

    def simulate_microsleep(self):
        self.logger.info("Simulating microsleep")
        if self.microsleep_active:
            self.logger.info("Microsleep already active, ignoring")
            return
        self.microsleep_active = True
        self.update_indicator("Microsleep Detection", "red")
        self.status_message.setText("⚠️ MICROSLEEP DETECTED! ⚠️")
        self.status_message.setStyleSheet("color: #e74c3c; font-size: 12px;")
        microsleep_thread = threading.Thread(target=self.microsleep_alert)
        microsleep_thread.daemon = True
        microsleep_thread.start()

    def toggle_eyes_indicator(self):
        if self.eyes_indicator_on:
            self.eyes_indicator.setColor("#2c3e50")
            self.eyes_indicator_on = False
        else:
            self.eyes_indicator.setColor("red")
            self.eyes_indicator_on = True

    def microsleep_alert(self):
        self.logger.info("Microsleep alert activated")
        try:
            cnt = 0
            orig_style = self.video_label.styleSheet()
            self.play_pre_recorded("MICROSLEEP! MICROSLEEP! MICROSLEEP!")
            while self.microsleep_active and cnt < 5:
                self.video_label.setStyleSheet("background-color: red;")
                time.sleep(0.5)
                self.video_label.setStyleSheet("background-color: black;")
                time.sleep(0.5)
                cnt += 1
            self.logger.info("Microsleep alert completed")
            self.microsleep_active = False
            self.video_label.setStyleSheet(orig_style)
            self.update_indicator("Microsleep Detection", "green")
            if self.is_monitoring:
                self.status_message.setText("Monitoring Active")
                self.status_message.setStyleSheet("color: #2ecc71; font-size: 12px;")
            else:
                self.status_message.setText("System Ready")
                self.status_message.setStyleSheet("color: #2ecc71; font-size: 12px;")
        except Exception as e:
            self.logger.error(f"Error in microsleep alert: {e}")
            self.microsleep_active = False

    def speak_alert(self, message):
        self.play_pre_recorded(message)

    def play_pre_recorded(self, message):
        with self._tts_lock:
            self.audio_manager.play_audio(message)

    def toggle_monitoring(self):
        self.logger.info(
            f"Toggling monitoring from {self.is_monitoring} to {not self.is_monitoring}"
        )
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.start_btn.setText("Stop Monitoring")
            self.start_btn.setStyleSheet(
                "background-color: #e74c3c; color: black; font-size: 12px; padding: 5px 10px; border: none; border-radius: 3px;"
            )
            self.status_message.setText("Monitoring Active")
            self.status_message.setStyleSheet("color: #2ecc71; font-size: 12px;")
            self.play_pre_recorded("Monitoring started")
        else:
            self.start_btn.setText("Start Monitoring")
            self.start_btn.setStyleSheet(
                "background-color: #2ecc71; color: black; font-size: 12px; padding: 5px 10px; border: none; border-radius: 3px;"
            )
            self.status_message.setText("System Ready")
            self.status_message.setStyleSheet("color: #2ecc71; font-size: 12px;")
            self.play_pre_recorded("Monitoring stopped")

    def update_ui(self):
        # This will eventually be used to refresh dynamic UI components.
        pass

    def show_error_message(self, message):
        self.error_label.setText(f"ERROR: {message}")
        self.logger.error(message)
        QTimer.singleShot(5000, lambda: self.error_label.setText(""))

    def update_indicator(self, name, status):
        try:
            if name in self.indicators:
                self.indicators[name].setColor(status)
        except Exception as e:
            self.logger.error(f"Error updating indicator {name}: {e}")

    def update_metric(self, name, value):
        try:
            if name in self.metric_values:
                self.metric_values[name].setText(value)
        except Exception as e:
            self.logger.error(f"Error updating metric {name}: {e}")

    def update_frame_display(self, frame):
        try:
            frame_resized = cv2.resize(frame, (320, 240))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        except Exception as e:
            self.logger.error(f"Error updating frame display: {e}")

    def processing_loop(self):
        self.logger.info("Entering processing loop")
        eyes_closed_frames = 0
        microsleep_threshold = 15
        blink_cnt = 0
        last_blink_time = time.time()
        while self.simulation_running:
            try:
                if not self.is_monitoring:
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            self.last_frame = frame.copy()
                            self.update_frame_display(frame)
                    time.sleep(0.05)
                    continue
                if not self.cap or not self.cap.isOpened():
                    self.logger.warning("Camera not available in processing loop")
                    time.sleep(0.5)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                self.last_frame = frame.copy()
                processed_frame, eyes_closed, confidence = self.process_eye_state(frame)
                self.update_frame_display(processed_frame)
                if eyes_closed:
                    eyes_closed_frames += 1
                    if (
                        eyes_closed_frames >= microsleep_threshold
                        and not self.microsleep_active
                    ):
                        self.simulate_microsleep()
                else:
                    if 1 <= eyes_closed_frames <= 5:
                        blink_cnt += 1
                    eyes_closed_frames = 0
                time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.5)
        self.logger.info("Exiting processing loop")

    def process_eye_state(self, frame):
        face_detected, left_eye, right_eye, vis_frame = (
            self.face_eye_detector.detect_face_and_eyes(frame)
        )
        if not face_detected:
            if len(self.eye_state_history) >= self.max_history:
                self.eye_state_history.pop(0)
            self.eye_state_history.append(False)
            closed_proportion = (
                sum(self.eye_state_history) / len(self.eye_state_history)
                if self.eye_state_history
                else 0
            )
            cv2.putText(
                vis_frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return vis_frame, closed_proportion > self.eyes_closed_threshold, 0.5

        left_eye_open, left_confidence = self.eye_state_detector.detect_eye_state(
            left_eye
        )
        right_eye_open, right_confidence = self.eye_state_detector.detect_eye_state(
            right_eye
        )
        eyes_closed = not (left_eye_open or right_eye_open)
        confidence = (left_confidence + right_confidence) / 2
        current_time = time.time()
        if self.last_eye_state is None:
            self.last_eye_state = eyes_closed
        if not self.last_eye_state and eyes_closed:
            self.blink_possible = True
            self.blink_timer_start = current_time
        if self.blink_possible and self.last_eye_state and not eyes_closed:
            if current_time - self.blink_timer_start < 0.5:
                self.blink_count += 1
            self.blink_possible = False

        self.last_eye_state = eyes_closed
        window_duration = current_time - self.blink_window_start
        blink_rate = (
            int(self.blink_count * 60 / window_duration) if window_duration > 0 else 0
        )

        if len(self.eye_state_history) >= self.max_history:
            self.eye_state_history.pop(0)
        self.eye_state_history.append(eyes_closed)
        closed_proportion = (
            sum(self.eye_state_history) / len(self.eye_state_history)
            if self.eye_state_history
            else 0
        )
        state_text = (
            "EYES CLOSED"
            if closed_proportion > self.eyes_closed_threshold
            else "EYES OPEN"
        )
        cv2.putText(
            vis_frame,
            f"{state_text} ({closed_proportion:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        self.update_metric("Blink Rate (per min):", str(blink_rate))

        return vis_frame, closed_proportion > self.eyes_closed_threshold, confidence

    def closeEvent(self, event):
        self.logger.info("Window closing, cleaning up resources")
        self.simulation_running = False
        if self.cap is not None:
            try:
                self.cap.release()
                self.logger.info("Camera released")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
        event.accept()

    def _update_countdown(self):
        remaining = int(self.next_pred - time.time())
        if remaining <= 0:
            if self.last_frame is not None:
                prob = predict_drowsiness_cnn(self.last_frame)
                if prob is not None:
                    pct = int(prob * 100)
                    self.update_metric("Drowsiness Probability:", f"{pct}%")
            # reset timer
            self.next_pred = time.time() + self.pred_interval
            remaining = self.pred_interval
        self.countdown_label.setText(f"Next update in: {remaining}s")


def main():
    try:
        app = QApplication(sys.argv)
        dashboard = DrowsinessDetectionDashboard()
        dashboard.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        try:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Fatal Error")
            error_dialog.setText("A fatal error occurred")
            error_dialog.setDetailedText(str(e))
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
        except Exception:
            pass


if __name__ == "__main__":
    main()
