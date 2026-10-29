"""
Odev 6. bolum: etiketlerin %10'unun yeniden etiketlenmesi ve tutarlilik olcumu.

1) python relabel_check.py sample
   -> pozitiflerin %10'unu (seed 42) FordLogo1/Project/relabel_10/ klasorune kopyalar.
   Bu klasoru birkac gun sonra Roboflow'da AYRI bir projeye yukleyip (eski kutulari
   gormeden) yeniden etiketleyin ve YOLO formatinda export edin.
   python relabel_check.py sample_extra 319 5
   -> sonradan eklenen pozitiflerden (ford_0319+) 5 ek gorsel: relabel_10_ek/
2) python relabel_check.py compare <ikinci_export_klasoru> [<diger_export> ...]
   -> iki etiketleme arasindaki kutulari IoU ile eslestirir; eksik/fazla kutu ve
   dusuk IoU'lu gorselleri listeler (rapordaki tutarsizliklar bolumu icin).
"""
import os
import sys
import random
import shutil
from PIL import Image
import imagehash
from common import DATASET, SPLITS, list_images, image_path, read_boxes, yolo_to_xyxy

OUT = os.path.abspath(os.path.join(DATASET, "..", "FordLogo1", "Project", "relabel_10"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def sample():
    pos = [(s, n) for s in SPLITS for n in list_images(s) if n.startswith("ford_")]
    picked = random.Random(42).sample(pos, max(1, round(len(pos) * 0.10)))
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for s, n in picked:
        shutil.copy2(image_path(s, n), OUT)
    print(f"{len(picked)} gorsel -> {OUT}")


def sample_extra(first, n):
    """sonradan eklenen pozitiflerden (ford_<first> ve sonrasi) ek ornek secer;
    toplam pozitif sayisi artinca %10 oranini korumak icin"""
    pos = [(s, f) for s in SPLITS for f in list_images(s)
           if f.startswith("ford_") and int(f[5:9]) >= first]
    picked = random.Random(42).sample(sorted(pos, key=lambda t: t[1]), n)
    out = OUT + "_ek"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for s, f in picked:
        shutil.copy2(image_path(s, f), out)
    print(f"{n} gorsel -> {out}: {sorted(f for _, f in picked)}")


def compare(export):
    orig = {}
    for s in SPLITS:
        for n in list_images(s):
            if n.startswith("ford_"):
                with Image.open(image_path(s, n)) as im:
                    orig[n] = (s, imagehash.phash(im))
    ious, issues = [], []
    n_img, unmatched = 0, 0
    for split in ("train", "valid", "test"):
        d = os.path.join(export, split, "images")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            with Image.open(os.path.join(d, f)) as im:
                h = imagehash.phash(im)
                W, H = im.size
            name, dist = min(((n, h - ph) for n, (s, ph) in orig.items()), key=lambda t: t[1])
            if dist > 4:
                continue
            s = orig[name][0]
            with Image.open(image_path(s, name)) as im:
                OW, OH = im.size
            a = [yolo_to_xyxy(b, OW, OH) for b in read_boxes(s, name)]
            lbl = os.path.join(export, split, "labels", os.path.splitext(f)[0] + ".txt")
            b = []
            for line in open(lbl):
                p = line.split()
                if len(p) == 5:
                    b.append(yolo_to_xyxy((0, *map(float, p[1:])), OW, OH))
            # greedy IoU eslestirme
            used, matched = set(), []
            for x in a:
                best = max(((iou(x, y), j) for j, y in enumerate(b) if j not in used), default=(0, None))
                if best[1] is not None and best[0] > 0.1:
                    used.add(best[1])
                    matched.append(best[0])
            ious += matched
            n_img += 1
            unmatched += len(a) - len(matched)
            if len(a) != len(b) or len(matched) < len(a) or any(v < 0.75 for v in matched):
                issues.append(f"{name}: ilk={len(a)} kutu, ikinci={len(b)} kutu, "
                              f"eslesmeyen={len(a) - len(matched)}, IoU={[round(v, 2) for v in matched]}")
    print(f"karsilastirilan gorsel: {n_img}, eslesmeyen kutu: {unmatched}")
    print(f"eslesen kutu: {len(ious)}, ortalama IoU: {sum(ious) / max(1, len(ious)):.3f}")
    print(f"IoU >= 0.75 orani: {sum(v >= 0.75 for v in ious) / max(1, len(ious)):.1%}")
    print("tutarsizliklar:")
    for i in issues:
        print("  -", i)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        sample()
    elif len(sys.argv) == 4 and sys.argv[1] == "sample_extra":
        sample_extra(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "compare":
        for export in sys.argv[2:]:
            print(f"== {export}")
            compare(export)
    else:
        print(__doc__)
