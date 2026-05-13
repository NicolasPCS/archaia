from transformers import pipeline
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
import shutil
import json
import os

# Argument parser
parser = argparse.ArgumentParser(description="Compute image classification")
parser.add_argument("input_path", type=str, help="Path to the input directory")

args = parser.parse_args()
input_path = args.input_path

checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification", use_fast=True)

# Setting up candidate labels for CLIP
candidate_labels = ["Archaeological artifact", "Documentation sheet", "Documentation sheet with archaeological artifact", "Archaeological excavation site", "Outdoor", "Landscape", "Archaeological structure", "Floor"]

# Define output .json file and target path for filtered images
output_file = Path(f"JSON_results/results_pred_dataset_{os.path.basename(input_path)}.json")
target_path = Path(f"/home/nicolascs/archaia_image-classification/filtered_images/{os.path.basename(input_path)}")
target_path.mkdir(parents=True, exist_ok=True)

target_path = str(target_path)

if output_file.exists():
    with output_file.open("r") as f:
        all_results = json.load(f)
else:
    all_results = []

files = sorted([f for f in os.listdir(input_path) if f.endswith(".jpg")])

"""
Classification
"""
fig = plt.figure(figsize=(9,13))

# Helper funtion to create plots
def add_image_subplot(rows, columns, subplot_idx, image, fname, prediction_score, prediction_label):
    image = image.convert('RGB')
    ax = fig.add_subplot(rows, columns, subplot_idx)
    ax.imshow(image)
    ax.set_title(f"{fname}\n{prediction_label}\n{prediction_score}", fontsize=9)
    ax.axis("off")

columns = 4
rows = 5

subplot_idx = 1
plt_cont = 1
cont = 0
cont_arch = 0

for fname in files:
    try:
        image_path = os.path.join(input_path, fname)

        # Load the image
        image = Image.open(image_path)

        predictions = detector(image, candidate_labels=candidate_labels)

        if subplot_idx == 21:
            plt.tight_layout()
            plt.savefig(f"Images2/{os.path.basename(input_path)}_CLIP_predictions_{(subplot_idx-1)*plt_cont}.png", dpi=300, bbox_inches="tight")

            fig = plt.figure(figsize=(9,13))

            plt_cont += 1
            subplot_idx = 1
        
        if subplot_idx < 21:
            if predictions[0]['label'] == "Archaeological artifact":
                add_image_subplot(rows, columns, subplot_idx, image, fname, predictions[0]['score'], predictions[0]['label'])

                cont_arch += 1
                subplot_idx += 1

        results = {
            "ground_truth:": str(fname),
            "prediction_score:": float(predictions[0]['score']),
            "prediction_label:": str(predictions[0]['label'])
        }

        all_results.append(results)

        cont += 1
    except:
        pass

number_of_images = {
    "Original number of images:": int(len(files)),
    "Number of images after filtering:": int(cont_arch)
}

all_results.append(number_of_images)

with output_file.open("w") as f:
    json.dump(all_results, f, indent=4)

# Save last 20 images
plt.tight_layout()
plt.savefig(f"Images2/{os.path.basename(input_path)}_CLIP_predictions_{(subplot_idx-1)*plt_cont}.png", dpi=300, bbox_inches="tight")

print(cont, "images appended to JSON file.")

"""
Move Images
"""
labeled_data = Path(f"JSON_results/results_pred_dataset_{os.path.basename(input_path)}.json")

lst_labeled_data = []
cont = 0

with labeled_data.open("r") as f:
    lst_labeled_data = json.load(f)

source_files = sorted([f for f in os.listdir(input_path) if f.endswith(".jpg")])

for i in range(len(lst_labeled_data)):
    if lst_labeled_data[i].get('prediction_label:') == "Archaeological artifact":
        source_file_path = os.path.join(input_path, lst_labeled_data[i]['ground_truth:'])
        cont += 1
        shutil.copy2(source_file_path, target_path)
    elif not lst_labeled_data[i].get('prediction_label:'):
        continue

print("Done!", cont)