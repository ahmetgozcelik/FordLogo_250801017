"""
Kalite kontrol: rastgele N gorselin kutularini cizer -> kalite_kontrol/
(odev: en az 30 rastgele gorsel)
"""
import shutil
import os
import random
import sys
from PIL import Image, ImageDraw
from common import DATASET, SPLITS, list_images, image_path, read_boxes, yolo_to_xyxy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = os.path.join(DATASET, "kalite_kontrol")
shutil.rmtree(OUT, ignore_errors=True)  # eski calistirmalardan kalanlari temizle
os.makedirs(OUT)

items = [(s, n) for s in SPLITS for n in list_images(s)]
rng = random.Random(42)
picked = rng.sample(items, N)

tiles = []
for s, name in picked:
    im = Image.open(image_path(s, name)).convert("RGB")
    d = ImageDraw.Draw(im)
    lw = max(2, min(im.size) // 150)
    for b in read_boxes(s, name):
        d.rectangle(yolo_to_xyxy(b, *im.size), outline=(255, 0, 0), width=lw)
    im.save(os.path.join(OUT, f"{s}_{name}"), quality=90)
    t = im.copy()
    t.thumbnail((256, 256))
    tiles.append((f"{s}/{name}", t))

# tek bir ozet pano (rapor icin)
cols = 8
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 256, rows * 272), "white")
d = ImageDraw.Draw(sheet)
for i, (label, t) in enumerate(tiles):
    x, y = (i % cols) * 256, (i // cols) * 272
    sheet.paste(t, (x + (256 - t.width) // 2, y + 16))
    d.text((x + 4, y + 2), label, fill=(0, 0, 160))
sheet.save(os.path.join(OUT, "_pano.jpg"), quality=85)
print(f"{N} gorsel -> {OUT}")
