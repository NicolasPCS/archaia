import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import supervision as sv
import matplotlib as plt
from pathlib import Path
from transformers import DetrForObjectDetection, DetrImageProcessor, AutoModelForImageSegmentation
from torchvision import transforms

# Import models
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "/home/nicolascs/archaia_image-classification/Object-Detection/finetuned_models/303artifacts_70epochs"

CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)

torch.set_float32_matmul_precision(["high", "highest"][0])

access_token = ""

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0", trust_remote_code=True, token=access_token#, low_cpu_mem_usage=False
)
birefnet.to("cuda")
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Helper functions
def make_detections(original_image):
    with torch.no_grad():
        # Load image and predict
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_TRESHOLD,
            target_sizes=target_sizes
        )[0]

    # Annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=IOU_TRESHOLD)

    labels = [
        f"{model.config.id2label[class_id]} {confidence:0.2f}" 
        for _, confidence, class_id, _ 
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=original_image.copy(), detections=detections, labels=labels)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return detections, frame

def crop_image_with_detections(original_image, detections):

    cropped_images = []

    for i in range(len(detections.class_id)):
        if detections.class_id[i] == 0:

            # Obtain bbox
            x1, y1, x2, y2 = detections.xyxy[i]

            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop
            cropped_image = original_image[y1:y2, x1:x2]

            cropped_images.append(cropped_image)
        
    return cropped_images

def fn(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb_image)
    original = im.copy()
    processed_image = process(im)
    processed_image = np.array(processed_image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2BGRA) # cv2 supports RGBA and BGRA
    return processed_image, original

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

parser = argparse.ArgumentParser(description="Compute object detection and BRIA")
parser.add_argument("input_path", type=str, help="Path to the input directory")
parser.add_argument("output_path", type=str, help="Path to the output directory")

args = parser.parse_args()

# Call
input_path = Path(args.input_path)
output_dir = Path(args.output_path)

output_dir.mkdir(parents=True, exist_ok=True)

files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(".jpg")])#[:10]

for fname in files:
    try:
        image_path = os.path.join(input_path, fname)
        original_image = cv2.imread(image_path)
        
        detections, frame = make_detections(original_image)
        cropped_images = crop_image_with_detections(original_image, detections)

        base_name = Path(fname).stem

        for j in range(len(cropped_images)):
            processed_image, original = fn(cropped_images[j])

            output_name = f"{base_name}_{j}.png"
            output_path = os.path.join(output_dir, output_name)

            cv2.imwrite(output_path, processed_image)
    except Exception as e:
        print("ERROR:", e)
        print("IMAGE PATH:",image_path)

        with open("error_log.txt", "a") as f:
            f.write(f"ERROR: {e}\n")
            f.write(f"IMAGE_PATH: {image_path}\n")
            f.write("-"*50 + "\n")

print("Done", len(files))