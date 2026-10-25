from pathlib import Path

from ultralytics import YOLO

from ..config import DEFAULT_PREDICTION_MODEL_PATH, PROJECT_ROOT

IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test" / "4_3690.jpg"

# DEFAULT_PREDICTION_MODEL_PATH : The model that has already been trained in google colab 

def predict_image(image_path=IMAGE_PATH, model_path=DEFAULT_PREDICTION_MODEL_PATH):
    model = YOLO(str(model_path))
    return model.predict(source=str(image_path), conf=0.5, imgsz=1024 ,save=True)


if __name__ == "__main__":
    predict_image()
