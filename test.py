from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco
import json

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="best.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

coco_gt = Coco.from_coco_dict_or_path("test_annotations.json")

mas = []
i = 0
for coco_image in coco_gt.images:
    result = get_sliced_prediction(
        coco_image.file_name,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_metric = 'IOU',
        postprocess_type = "NMS",
        postprocess_match_threshold = 0.6,
        
    )
    result.export_visuals(
        export_dir="demo_data/",
        file_name=coco_image.file_name.split("/")[-1],
        rect_th=2,
    )
    i += 1
    for pred in result.object_prediction_list:
        curr_dict = {
            "image_id": coco_image.id,
            "category_id": 0,
            "bbox": pred.bbox.to_coco_bbox(),
            "score": pred.score.value,
        }
        print(curr_dict, pred.bbox.to_xywh())
        mas.append(curr_dict)


with open('predictions.json', 'w') as f:
    json.dump(mas, f)

print("sahi coco evaluate --dataset_json_path test_annotations.json --result_json_path predictions.json --out_dir results")
