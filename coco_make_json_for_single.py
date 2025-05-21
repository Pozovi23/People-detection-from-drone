import os
import json

images_dir = "dataset/test_images"
labels_dir = "dataset/test_labels"
output_json = "test_annotations.json"

categories = [{"id": 0, "name": "person"}]

coco_data = {
    "images": [],
    "annotations": [],
    "categories": categories
}

annotation_id = 1

for image_id, image_name in [(0, "train_ZRI_2070_JPG.rf.826eb42eeed0ddf7d242a0bc0436f95a.jpg")]:
    if not image_name.endswith((".jpg", ".png")):
        continue
    
    image_path = os.path.join(images_dir, image_name)

    coco_data["images"].append({
        "id": image_id,
        "file_name": "dataset/test_images/" + image_name,
        "width": 4000,
        "height": 3000,
    })

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)
    
    if not os.path.exists(label_path):
        continue
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())

        x = (x_center - w / 2)
        y = (y_center - h / 2)
        
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(class_id),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })
        annotation_id += 1

with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=2)
