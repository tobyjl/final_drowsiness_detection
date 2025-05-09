import os

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image


class DrowsinessDetectionCNN(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {
                "num_classes": 2,
                "dropout_rate": 0.7,
                "input_channels": 1,
                "input_height": 56,
                "input_width": 224,
            }
        logger.debug("Initialising CNN architecture with config: {}", config)
        self.conv1 = torch.nn.Conv2d(config["input_channels"], 8, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.MaxPool2d(2, 2)
        fh = config["input_height"] // 8
        fw = config["input_width"] // 8
        self.feature_size = 32 * fh * fw
        self.fc1 = torch.nn.Linear(self.feature_size, 64)
        self.bn4 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bn5 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, config["num_classes"])
        self.dropout = torch.nn.Dropout(config["dropout_rate"])
        logger.info("CNN layers initialised; feature_size={}", self.feature_size)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.dropout(F.relu(self.bn5(self.fc2(x))))
        return self.fc3(x)


_device = torch.device("cpu")
logger.info("Using device: {}", _device)

_model = DrowsinessDetectionCNN().to(_device)
_ckpt_path = os.path.join(
    os.path.dirname(__file__), "outputs/models/deep_ml/TJM3_FINAL_20250504_230729.pt"
)
logger.info("Loading checkpoint from {}", _ckpt_path)
_checkpoint = torch.load(_ckpt_path, map_location=_device)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.eval()
logger.success("Model loaded and set to eval mode")

_val_transform = Compose(
    [
        Resize(56, 224),
        ToTensorV2(),
    ]
)
logger.debug("Validation transform: {}", _val_transform)


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = img.convert("L")
        arr = np.array(arr)
        out = self.transform(image=arr)["image"]
        return out.float()


_val_aug = AlbumentationsTransform(_val_transform)

_face_detector = dlib.get_frontal_face_detector()
logger.info("Face detector initialised")


def _crop_face(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(gray, 1)
    if not faces:
        logger.warning("No face detected in frame")
        return None
    f = faces[0]
    x1, y1, x2, y2 = f.left(), f.top(), f.right(), f.bottom()
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    logger.debug(
        "Cropping face region: x1={}, y1={}, x2={}, y2={}, frame_shape={}",
        x1,
        y1,
        x2,
        y2,
        frame_bgr.shape,
    )
    return frame_bgr[y1:y2, x1:x2]


def predict_drowsiness_cnn(frame_bgr):
    logger.debug("Received frame for prediction, shape={}", frame_bgr.shape)
    face = _crop_face(frame_bgr)
    if face is None:
        logger.info("Skipping prediction (no face)")
        return None

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)

    try:
        tensor = _val_aug(pil).unsqueeze(0).to(_device)
    except Exception as e:
        logger.error("Transform error: {}", e)
        return None

    mn, mx = tensor.min().item(), tensor.max().item()
    logger.debug(
        "Input tensor stats: min={:.4f}, max={:.4f}, shape={}", mn, mx, tensor.shape
    )

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)

    drowsy_prob = probs[0, 1].item()
    logger.info("Predicted drowsiness probability: {:.3f}", drowsy_prob)
    return drowsy_prob
