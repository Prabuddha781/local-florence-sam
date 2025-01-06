import os
from typing import Tuple, Optional

import time
import logging
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
from utils.video import generate_unique_name, create_directory, delete_directory

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, \
    IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
from utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model

VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    output_image.save("output.png")
    return output_image


def process_image(
    image_input, text_input
) -> Tuple[Optional[Image.Image], Optional[str]]:
    if not text_input:
        logging.error("No text input.")
        return
    
    texts = [prompt.strip() for prompt in text_input.split(",")]

    start = time.time()
    detections_list = []
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        print(f"Time taken for florence: {time.time() - start}")
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        detections_list.append(detections)
        print(f"Time taken for sam: {time.time() - start}")
    
    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
    return annotate_image(image_input, detections), None


if __name__ == "__main__":
    image = Image.open("img.png").convert("RGB")
    text = "human, road, sidewalk, car, grass, fence"

    video_path = "vid.MOV"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame_rgb)

        img, _ = process_image(frame, text)
        if img:
            img.show()