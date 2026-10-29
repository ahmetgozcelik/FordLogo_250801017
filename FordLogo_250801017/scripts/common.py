"""scriptlerin ortak yardimcilari"""
import os

DATASET = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPLITS = ("train", "val", "test")


def list_images(split):
    d = os.path.join(DATASET, "images", split)
    return sorted(f for f in os.listdir(d) if f.endswith(".jpg"))


def image_path(split, name):
    return os.path.join(DATASET, "images", split, name)


def label_path(split, name):
    return os.path.join(DATASET, "labels", split, os.path.splitext(name)[0] + ".txt")


def read_boxes(split, name):
    """YOLO etiketlerini [(cls, xc, yc, w, h), ...] olarak dondurur"""
    with open(label_path(split, name)) as f:
        return [(int(p[0]), *map(float, p[1:])) for p in (l.split() for l in f) if p]


def yolo_to_xyxy(box, w, h):
    _, xc, yc, bw, bh = box
    return ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
