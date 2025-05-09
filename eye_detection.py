import os

import cv2
import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torchvision.models import resnet18


class EyeStateDetector:
    def __init__(
        self, model_path="outputs/models/resnet/best_eyes_model.pth", device=None
    ):
        self.logger = logger
        self.logger.info("Initialising Eye State Detector")
        self.logger.info(f"Looking for model at path: {model_path}")
        self.logger.info(f"Absolute path: {os.path.abspath(model_path)}")
        self.logger.info(f"File exists: {os.path.exists(model_path)}")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.logger.info(f"Using device: {self.device}")
        try:

            class CustomResNet(torch.nn.Module):
                def __init__(self):
                    super(CustomResNet, self).__init__()
                    self.backbone = resnet18(pretrained=False)
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = torch.nn.Linear(in_features, 2)

                def forward(self, x):
                    return self.backbone(x)

            self.logger.info("Creating custom model with backbone structure")
            self.model = CustomResNet()
            state_dict = torch.load(model_path, map_location=self.device)
            self.logger.info(f"Loaded state dict with {len(state_dict)} keys")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = (
                    key.replace("backbone.", "") if key.startswith("backbone.") else key
                )
                new_state_dict[new_key] = value
            self.logger.info("Processed state dict keys to remove 'backbone.' prefix")
            self.model.backbone.load_state_dict(new_state_dict, strict=True)
            self.logger.info("Successfully loaded state dict into the backbone")
            self.model.eval()
            self.model.to(self.device)
            self.logger.info("Model setup completed successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # resize image to model input dimensions
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect_eye_state(self, eye_image):
        if self.model is None:
            self.logger.warning("Model not loaded, returning default values")
            return True, 0.5
        try:
            if eye_image is None or eye_image.size == 0:
                return True, 0.5
            eye_pil = Image.fromarray(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB))
            eye_tensor = self.transform(eye_pil).unsqueeze(0)
            eye_tensor = eye_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(eye_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            is_open = probabilities[0, 1] > probabilities[0, 0]
            confidence = (
                probabilities[0, 1].item() if is_open else probabilities[0, 0].item()
            )
            return is_open, confidence
        except Exception as e:
            self.logger.error(f"Error (detect_eye_state): {e}")
            return True, 0.5


class FaceEyeDetector:
    def __init__(self):
        self.logger = logger
        self.logger.info("Initialising Face and Eye Detector")
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            self.logger.info("Face and eye detectors loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading detectors: {e}")
            self.face_cascade = None
            self.eye_cascade = None

    def detect_face_and_eyes(self, frame):
        if self.face_cascade is None or self.eye_cascade is None:
            self.logger.warning("Detectors not loaded, returning default values")
            return False, None, None, frame
        try:
            vis_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return False, None, None, vis_frame
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_gray = gray[y : y + h, x : x + w]
            face_color = frame[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(face_gray)
            if len(eyes) < 2:
                eye_w = w // 4
                eye_h = h // 6
                left_eye_x = x + w // 4
                left_eye_y = y + h // 4
                left_eye = frame[
                    left_eye_y : left_eye_y + eye_h, left_eye_x : left_eye_x + eye_w
                ]
                right_eye_x = x + w // 2
                right_eye_y = y + h // 4
                right_eye = frame[
                    right_eye_y : right_eye_y + eye_h, right_eye_x : right_eye_x + eye_w
                ]
                cv2.rectangle(
                    vis_frame,
                    (left_eye_x, left_eye_y),
                    (left_eye_x + eye_w, left_eye_y + eye_h),
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    vis_frame,
                    (right_eye_x, right_eye_y),
                    (right_eye_x + eye_w, right_eye_y + eye_h),
                    (255, 0, 0),
                    2,
                )
            else:
                eyes = sorted(eyes, key=lambda e: e[0])
                eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
                eye2_x, eye2_y, eye2_w, eye2_h = eyes[-1]
                left_eye = face_color[
                    eye1_y : eye1_y + eye1_h, eye1_x : eye1_x + eye1_w
                ]
                right_eye = face_color[
                    eye2_y : eye2_y + eye2_h, eye2_x : eye2_x + eye2_w
                ]
                cv2.rectangle(
                    vis_frame,
                    (x + eye1_x, y + eye1_y),
                    (x + eye1_x + eye1_w, y + eye1_y + eye1_h),
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    vis_frame,
                    (x + eye2_x, y + eye2_y),
                    (x + eye2_x + eye2_w, y + eye2_y + eye2_h),
                    (255, 0, 0),
                    2,
                )
            return True, left_eye, right_eye, vis_frame
        except Exception as e:
            self.logger.error(f"Error in detect_face_and_eyes: {e}")
            return False, None, None, vis_frame
