"""
YOLO etiketlerini COCO JSON'a donusturur -> annotations/instances_{train,val,test}.json
YOLO : <sinif> <x_merkez> <y_merkez> <genislik> <yukseklik>  (0-1 normalize, merkez)
COCO : bbox = [x_sol_ust, y_sol_ust, genislik, yukseklik]     (piksel, sol ust kose)
"""
import os
import json
from PIL import Image
from common import DATASET, SPLITS, list_images, image_path, read_boxes

OUT = os.path.join(DATASET, "annotations")
os.makedirs(OUT, exist_ok=True)
ann_id = 1
for s in SPLITS:
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "ford_logo", "supercategory": "logo"}]}
    for img_id, name in enumerate(list_images(s), 1):
        with Image.open(image_path(s, name)) as im:
            W, H = im.size
        coco["images"].append({"id": img_id, "file_name": name, "width": W, "height": H})
        for _, xc, yc, w, h in read_boxes(s, name):
            bw, bh = w * W, h * H
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(xc * W - bw / 2, 2), round(yc * H - bh / 2, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
            })
            ann_id += 1
    with open(os.path.join(OUT, f"instances_{s}.json"), "w") as f:
        json.dump(coco, f)
    print(f"{s}: {len(coco['images'])} gorsel, {len(coco['annotations'])} kutu")
