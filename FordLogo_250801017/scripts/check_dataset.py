"""
Veri seti dogrulama scripti (Odev 2, bolum 5 ve 6'daki zorunlu format kurallari).

Ders kapsaminda resmi bir check_dataset.py paylasilmadigi icin odev yonergesinde
listelenen tum kontroller burada uygulanmistir. Bagimsiz calisir (yalnizca Pillow
gerekir), baska veri setlerinde de kullanilabilir:

    python check_dataset.py                  # scripts/ klasorunun bir ustu
    python check_dataset.py /yol/FordLogo_No

HATA  : yonergeye aykiri durum (cikis kodu 1)
UYARI : kural ihlali olmayan ama gozden gecirilmesi onerilen durum
"""
import os
import re
import sys
import csv
import hashlib
from PIL import Image

SPLITS = ("train", "val", "test")
CLASS_NAME = "ford_logo"
META_FIELDS = ["file_name", "split", "has_logo", "num_boxes", "width", "height", "source_url", "notes"]
NAME_RE = re.compile(r"^(ford|neg)_\d{4}\.jpg$")
MIN_SIDE = 320
MIN_BOX_PX = 10
EPS = 1e-6

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def check_structure(root):
    for sub in ("images", "labels"):
        for s in SPLITS:
            if not os.path.isdir(os.path.join(root, sub, s)):
                err(f"klasor yok: {sub}/{s}")
    for f in ("data.yaml", "classes.txt", "metadata.csv", "README.md", "scripts/split.py"):
        if not os.path.isfile(os.path.join(root, f)):
            err(f"dosya yok: {f}")
    if not os.path.isdir(os.path.join(root, "augmentation_ornekleri")):
        err("klasor yok: augmentation_ornekleri/")


def check_yaml(root):
    path = os.path.join(root, "data.yaml")
    if not os.path.isfile(path):
        return
    cfg = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    for s in SPLITS:
        if cfg.get(s) != f"images/{s}":
            err(f"data.yaml: '{s}' degeri 'images/{s}' olmali (bulunan: {cfg.get(s)!r})")
    if cfg.get("nc") != "1":
        err(f"data.yaml: nc 1 olmali (bulunan: {cfg.get('nc')!r})")
    names = re.findall(r"[\"']?([\w-]+)[\"']?", cfg.get("names", ""))
    if names != [CLASS_NAME]:
        err(f"data.yaml: names [\"{CLASS_NAME}\"] olmali (bulunan: {cfg.get('names')!r})")

    path = os.path.join(root, "classes.txt")
    if os.path.isfile(path):
        lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        if lines != [CLASS_NAME]:
            err(f"classes.txt tek satir '{CLASS_NAME}' olmali (bulunan: {lines})")


def check_image(path, rel):
    """gorsel bilgisi (boyut) dondurur; okunamazsa None"""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if im.format != "JPEG":
                err(f"{rel}: JPEG degil ({im.format})")
            if im.mode != "RGB":
                err(f"{rel}: RGB degil ({im.mode})")
            if im.getexif().get(0x0112, 1) != 1:
                err(f"{rel}: EXIF orientation uygulanmamis")
            if min(im.size) < MIN_SIDE:
                err(f"{rel}: kisa kenar {min(im.size)} px < {MIN_SIDE}")
            return im.size
    except Exception as e:
        err(f"{rel}: gorsel okunamadi ({e})")
        return None


def check_label(path, rel, size, is_neg):
    """kutu sayisini dondurur"""
    raw = open(path, encoding="utf-8").read()
    lines = [l for l in raw.splitlines() if l.strip()]
    if is_neg and lines:
        err(f"{rel}: negatif gorselin etiketi bos olmali ({len(lines)} kutu var)")
    if not is_neg and not lines:
        err(f"{rel}: pozitif gorselin etiketi bos")
    for i, l in enumerate(lines, 1):
        p = l.split()
        if len(p) != 5:
            err(f"{rel}:{i}: 5 deger olmali -> {l!r}")
            continue
        if p[0] != "0":
            err(f"{rel}:{i}: gecersiz sinif no {p[0]!r} (yalnizca 0)")
        try:
            xc, yc, w, h = map(float, p[1:])
        except ValueError:
            err(f"{rel}:{i}: sayi olmayan deger -> {l!r}")
            continue
        if not all(0 <= v <= 1 for v in (xc, yc, w, h)):
            err(f"{rel}:{i}: 0-1 araligi disinda -> {l!r}")
            continue
        if w <= 0 or h <= 0:
            err(f"{rel}:{i}: genislik/yukseklik 0 -> {l!r}")
            continue
        if xc - w / 2 < -EPS or xc + w / 2 > 1 + EPS or yc - h / 2 < -EPS or yc + h / 2 > 1 + EPS:
            err(f"{rel}:{i}: kutu gorsel disina tasiyor -> {l!r}")
        if size and min(w * size[0], h * size[1]) < MIN_BOX_PX:
            warn(f"{rel}:{i}: kutu kisa kenari {MIN_BOX_PX} px'den kucuk")
    return len(lines)


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    print(f"veri seti: {root}")
    check_structure(root)
    check_yaml(root)

    found = {}    # file_name -> (split, size, num_boxes)
    digests = {}  # md5 -> (split, file_name)
    for s in SPLITS:
        img_dir = os.path.join(root, "images", s)
        lbl_dir = os.path.join(root, "labels", s)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if not f.startswith("."))
        lbls = sorted(f for f in os.listdir(lbl_dir) if not f.startswith("."))
        stems = {os.path.splitext(f)[0] for f in imgs}
        for f in lbls:
            if not f.endswith(".txt"):
                err(f"labels/{s}/{f}: .txt degil")
            elif os.path.splitext(f)[0] not in stems:
                err(f"labels/{s}/{f}: karsilik gelen gorsel yok")
        for f in imgs:
            rel = f"images/{s}/{f}"
            if not NAME_RE.match(f):
                err(f"{rel}: dosya adi ford_0001.jpg / neg_0001.jpg formatinda degil")
            if f in found:
                err(f"{f}: birden fazla bolumde ({found[f][0]} ve {s})")
            md5 = hashlib.md5(open(os.path.join(img_dir, f), "rb").read()).hexdigest()
            if md5 in digests:
                err(f"{rel}: {digests[md5][0]}/{digests[md5][1]} ile birebir ayni dosya")
            digests[md5] = (s, f)
            size = check_image(os.path.join(img_dir, f), rel)
            lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
            if not os.path.isfile(lbl):
                err(f"{rel}: etiket dosyasi yok (labels/{s}/{os.path.splitext(f)[0]}.txt)")
                found[f] = (s, size, None)
                continue
            n = check_label(lbl, f"labels/{s}/{os.path.splitext(f)[0]}.txt", size, f.startswith("neg_"))
            found[f] = (s, size, n)

    # metadata.csv
    path = os.path.join(root, "metadata.csv")
    if os.path.isfile(path):
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != META_FIELDS:
                err(f"metadata.csv sutunlari {','.join(META_FIELDS)} olmali (bulunan: {reader.fieldnames})")
            rows = list(reader)
        listed = set()
        for i, r in enumerate(rows, 2):
            n = r.get("file_name", "")
            if n in listed:
                err(f"metadata.csv:{i}: {n} birden fazla satirda")
            listed.add(n)
            if n not in found:
                err(f"metadata.csv:{i}: {n} veri setinde yok")
                continue
            s, size, nb = found[n]
            if r.get("split") != s:
                err(f"metadata.csv:{i}: {n} split={r.get('split')!r}, dosya {s} klasorunde")
            if nb is not None:
                if r.get("num_boxes") != str(nb):
                    err(f"metadata.csv:{i}: {n} num_boxes={r.get('num_boxes')!r}, etikette {nb} kutu")
                if r.get("has_logo") != str(int(nb > 0)):
                    err(f"metadata.csv:{i}: {n} has_logo={r.get('has_logo')!r} kutu sayisiyla tutarsiz")
            if r.get("has_logo") != ("1" if n.startswith("ford_") else "0"):
                err(f"metadata.csv:{i}: {n} has_logo dosya adi onekiyle tutarsiz")
            if size and (r.get("width"), r.get("height")) != (str(size[0]), str(size[1])):
                err(f"metadata.csv:{i}: {n} boyut {r.get('width')}x{r.get('height')}, "
                    f"gorsel {size[0]}x{size[1]}")
            if not r.get("source_url", "").startswith("http"):
                err(f"metadata.csv:{i}: {n} source_url gecersiz/bos")
        for n in sorted(set(found) - listed):
            err(f"metadata.csv: {n} icin satir yok")

    # ozet
    total = len(found)
    if total:
        print(f"\n{'bolum':6s} {'gorsel':>7s} {'pozitif':>8s} {'negatif':>8s} {'kutu':>6s} {'oran':>6s}")
        for s in SPLITS:
            part = [v for k, v in found.items() if v[0] == s]
            pos = sum(1 for k, v in found.items() if v[0] == s and k.startswith("ford_"))
            boxes = sum(v[2] or 0 for v in part)
            print(f"{s:6s} {len(part):7d} {pos:8d} {len(part) - pos:8d} {boxes:6d} {len(part) / total:6.1%}")

    for w in warnings:
        print("UYARI:", w)
    for e in errors:
        print("HATA :", e)
    print(f"\n{total} gorsel kontrol edildi: {len(errors)} hata, {len(warnings)} uyari.")
    print("SONUC:", "BASARISIZ" if errors else "BASARILI")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
