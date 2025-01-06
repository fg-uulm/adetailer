from __future__ import annotations

from pathlib import Path
import pprint
from typing import TYPE_CHECKING

import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox, FaceDataList, FaceData, Region, Gender, Race, Emotion

if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld
    from deepface import DeepFace


def ultralytics_predict(
    model_path: str | Path,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
    classes: str = "",
) -> PredictOutput[float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    apply_classes(model, model_path, classes)
    pred = model(image, conf=confidence, device=device)
    
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return PredictOutput()
    
    bboxes = bboxes.tolist()    

    confs = pred[0].boxes.data[:, 4:6].cpu().numpy().tolist()
    pprint.pp(confs)

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    confidences = pred[0].boxes.conf.cpu().numpy().tolist()

    preview = pred[0].plot()
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    preview = Image.fromarray(preview)

    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview, confs=confs)

def metadata_predict(
    image: Image.Image,
) -> FaceDataList:
    from deepface import DeepFace
    import numpy as np
    
    # convert PIL image to BGR numpy array
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # analyze the image    
    res = DeepFace.analyze(image, actions=["age","gender","emotion","race"], enforce_detection=False, detector_backend="yolov8", expand_percentage=10, silent=True)
    # convert result to FaceDataList
    face_data = FaceDataList()
    # Preliminary data processing, will be done again after filtering in adetailer
    # first calculate avg age, max age, min age
    ages = [face["age"] for face in res]
    median_age = np.average(ages)
    max_age = max(ages)
    min_age = min(ages)
    # calculate median gender from Man/Woman floats (100% in total per face) of all faces
    genders = [face["gender"]["Woman"] for face in res]
    median_gender = np.average(genders)
    max_gender = max(genders)
    min_gender = min(genders)
    
    for face in res:
        face_data.faces.append(FaceData(
            age = face["age"],
            median_age=median_age,
            median_gender=median_gender,
            max_age=max_age,
            min_age=min_age,
            max_gender=max_gender,
            min_gender=min_gender,
            region = Region(
                x = face["region"]["x"],
                y = face["region"]["y"],
                w = face["region"]["w"],
                h = face["region"]["h"],
                left_eye = face["region"]["left_eye"],
                right_eye = face["region"]["right_eye"]
            ),
            face_confidence = face["face_confidence"],
            gender=Gender(
                Woman=face["gender"]["Woman"],
                Man=face["gender"]["Man"]
            ),
            dominant_gender=face["dominant_gender"],
            race=Race(
                asian=face["race"]["asian"],
                indian=face["race"]["indian"],
                black=face["race"]["black"],
                white=face["race"]["white"],
                middle_eastern=face["race"]["middle eastern"],
                latino_hispanic=face["race"]["latino hispanic"]
            ),
            dominant_race=face["dominant_race"],
            emotion=Emotion(
                angry=face["emotion"]["angry"],
                disgust=face["emotion"]["disgust"],
                fear=face["emotion"]["fear"],
                happy=face["emotion"]["happy"],
                sad=face["emotion"]["sad"],
                surprise=face["emotion"]["surprise"],
                neutral=face["emotion"]["neutral"]
            ),
            dominant_emotion=face["dominant_emotion"]
        )
    )
    return face_data

def apply_classes(model: YOLO | YOLOWorld, model_path: str | Path, classes: str):
    if not classes or "-world" not in Path(model_path).stem:
        return
    parsed = [c.strip() for c in classes.split(",") if c.strip()]
    if parsed:
        model.set_classes(parsed)


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (W, H) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]
