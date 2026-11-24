
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from PIL import Image
import numpy as np
import cv2


# Load the model only once
_model = None


def load_damage_model():
    global _model

    if _model is None:
        weights_path = hf_hub_download(
            repo_id="shanexf/car-damage-yolo11m-ft",
            filename="damage_best.pt"
        )

        _model = YOLO(weights_path)

    return _model


def detect_damage(image, confidence=0.25):

    model = load_damage_model()

    # Convert uploaded image into OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    elif isinstance(image, str):
        image = cv2.imread(image)

        if image is None:
            raise ValueError("Could not read the image.")

    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must have three color channels.")

    else:
        raise TypeError("Unsupported image format.")

    # Run YOLO detection
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # Generate annotated image
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    detections = []
    damage_counts = {}

    # Extract detected damage information
    for box in result.boxes:

        class_id = int(box.cls[0])
        damage_name = model.names[class_id]

        score = float(box.conf[0])

        coordinates = box.xyxy[0].tolist()

        detections.append({
            "damage": damage_name,
            "confidence": round(score * 100, 2),
            "bounding_box": [
                round(value, 2) for value in coordinates
            ]
        })

        damage_counts[damage_name] = (
            damage_counts.get(damage_name, 0) + 1
        )

    return {
        "annotated_image": annotated_image,
        "detections": detections,
        "damage_counts": damage_counts,
        "total_damages": len(detections)
    }
