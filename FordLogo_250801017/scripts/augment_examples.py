"""
Veri artirma ornekleri (albumentations): kutular gorselle birlikte donusturulur.
Cikti yalnizca rapor icindir -> augmentation_ornekleri/ (train/val/test'e KARISTIRILMAZ)
"""
import shutil
import os
import random
import numpy as np
import albumentations as A
from PIL import Image, ImageDraw
from common import DATASET, list_images, image_path, read_boxes

OUT = os.path.join(DATASET, "augmentation_ornekleri")
shutil.rmtree(OUT, ignore_errors=True)  # eski calistirmalardan kalanlari temizle
os.makedirs(OUT)
bbox = A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.3)

TRANSFORMS = {
    "yatay_cevirme": A.HorizontalFlip(p=1),
    "dondurme_15": A.Affine(rotate=(-15, 15), p=1),
    "parlaklik_kontrast": A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1),
    "olcekleme": A.Affine(scale=(0.6, 1.3), p=1),
    "rastgele_kirpma": A.RandomSizedBBoxSafeCrop(height=512, width=512, p=1),
    "renk_tonu": A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=1),
    "bulaniklik": A.GaussianBlur(blur_limit=(5, 9), p=1),
    "karma": [A.HorizontalFlip(p=0.5), A.Affine(scale=(0.7, 1.2), rotate=(-10, 10), p=1),
              A.RandomBrightnessContrast(p=1), A.GaussNoise(p=0.5)],
}


def draw(arr, boxes):
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    W, H = im.size
    for xc, yc, w, h in boxes:
        d.rectangle(((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H),
                    outline=(255, 0, 0), width=max(2, min(W, H) // 150))
    im.thumbnail((320, 320))
    return im


random.seed(42)
np.random.seed(42)
names = random.Random(42).sample([n for n in list_images("train") if n.startswith("ford_")], 4)
for name in names:
    img = np.array(Image.open(image_path("train", name)).convert("RGB"))
    boxes = [b[1:] for b in read_boxes("train", name)]
    tiles = [("orijinal", draw(img, boxes))]
    for tname, t in TRANSFORMS.items():
        aug = A.Compose(t if isinstance(t, list) else [t], bbox_params=bbox, seed=42)
        r = aug(image=img, bboxes=boxes, cls=[0] * len(boxes))
        tiles.append((tname, draw(r["image"], r["bboxes"])))
    sheet = Image.new("RGB", (3 * 330, 3 * 350), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(tiles):
        x, y = (i % 3) * 330, (i // 3) * 350
        sheet.paste(t, (x + 5, y + 25))
        d.text((x + 5, y + 5), label, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, f"aug_{name}"), quality=88)
    print("->", f"aug_{name}")
