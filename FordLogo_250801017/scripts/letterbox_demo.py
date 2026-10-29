"""
Ayri deneme: oran korunarak 224x224 letterbox yeniden boyutlandirma.
Kutular ayni olcek + kaydirma ile guncellenir; once/sonra gorselleri -> letterbox_ornekleri/
(Teslim edilen veri setindeki gorseller orijinal boyutta kalir.)
"""
import shutil
import os
import random
from PIL import Image, ImageDraw
from common import DATASET, list_images, image_path, read_boxes, yolo_to_xyxy

SIZE = 224
PAD = (114, 114, 114)
OUT = os.path.join(DATASET, "letterbox_ornekleri")
shutil.rmtree(OUT, ignore_errors=True)  # eski calistirmalardan kalanlari temizle
os.makedirs(OUT)


def letterbox(im, boxes, size=SIZE):
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), PAD)
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (dx, dy))
    new_boxes = []
    for c, xc, yc, bw, bh in boxes:
        # piksel -> olcekle -> kaydir -> yeni boyuta gore normalize et
        new_boxes.append((c, (xc * w * r + dx) / size, (yc * h * r + dy) / size,
                          bw * w * r / size, bh * h * r / size))
    return canvas, new_boxes


def draw(im, boxes, lw):
    im = im.copy()
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(yolo_to_xyxy(b, *im.size), outline=(255, 0, 0), width=lw)
    return im


rng = random.Random(42)
names = rng.sample([n for n in list_images("train") if n.startswith("ford_")], 6)
rows = []
for name in names:
    im = Image.open(image_path("train", name)).convert("RGB")
    boxes = read_boxes("train", name)
    lb, lb_boxes = letterbox(im, boxes)
    before = draw(im, boxes, max(2, min(im.size) // 150))
    before.thumbnail((448, 448))
    after = draw(lb, lb_boxes, 2).resize((448, 448), Image.NEAREST)
    row = Image.new("RGB", (448 * 2 + 16, 448), "white")
    row.paste(before, ((448 - before.width) // 2, (448 - before.height) // 2))
    row.paste(after, (448 + 16, 0))
    row.save(os.path.join(OUT, f"letterbox_{name}"), quality=90)
    rows.append(row)
    print(f"{name}: {im.size} -> {SIZE}x{SIZE}  ornek kutu {boxes[0][1:]} -> "
          f"{tuple(round(v, 4) for v in lb_boxes[0][1:])}")

sheet = Image.new("RGB", (rows[0].width, sum(r.height + 8 for r in rows)), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height + 8
sheet.save(os.path.join(OUT, "_pano.jpg"), quality=85)
