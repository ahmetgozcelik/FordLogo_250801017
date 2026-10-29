"""
Roboflow YOLO export(lar)indaki etiketleri, preprocess.py ile uretilen
orijinal boyutlu gorsellere (processed_data) aktarir ve etiketli havuzu olusturur.

Neden gerekli: Roboflow'a processed_data yerine raw_data yuklendigi icin export'ta
- dosya adlari 000001_jpg.rf.<hash>.jpg seklinde,
- preprocess.py'nin eledigi kucuk (<320px) ve tekrar eden gorseller tekrar mevcut.
Bu script her export gorselini perceptual hash ile processed_data'daki karsiligina
eslestirir. YOLO etiketleri normalize oldugundan, en-boy orani ayni olan gorsellere
dogrudan aktarilabilir.

Cikti: Project/labeled_pool/{images,labels}/ + pool.csv  (split.py bunu kullanir)
       Project/to_label/  -> henuz etiketlenmemis / kontrol gereken gorseller

Kullanim (FordLogo_<No>/scripts klasorunden):
    python build_dataset.py
Yeni bir Roboflow export'u eklendiginde EXPORTS listesine ekleyip tekrar calistirin
(listede sonra gelen export, onceki etiketlerin yerine gecer).
"""
import os
import csv
import shutil
from PIL import Image
import imagehash

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "FordLogo1"))
PROJECT = os.path.join(SRC_ROOT, "Project")
PROCESSED = os.path.join(PROJECT, "processed_data")
POOL = os.path.join(PROJECT, "labeled_pool")
TO_LABEL = os.path.join(PROJECT, "to_label")

# eskiden yeniye Roboflow export klasorleri (train/valid/test alt klasorlu)
EXPORTS = [
    SRC_ROOT,                                   # FordLogo1 v1 export
    os.path.join(SRC_ROOT, "..", "fordlogo"),       # ikinci etiketleme turu (to_label)
    os.path.join(SRC_ROOT, "..", "fordlogo3"),      # ucuncu tur: kucuk/uzak logolar (download_small_logos.py)
    os.path.join(SRC_ROOT, "..", "fordlogo7"),      # eski armalar: ozel karar 3 geregi yalnizca yazi (15 gorsel)
]
# download_small_logos.py ile eklenen gorseller bu numaradan baslar (processed_data/positive)
SMALL_LOGO_START = 400

# negative klasorunde olup Ford logosu iceren / icerebilen gorseller.
# Etiketlenene kadar havuza alinmaz; etiketlenirse kutusu varsa pozitife,
# kutusuz export edilirse (logo yok kararı) negatife gecer.
NEG_NEEDS_REVIEW = {
    "neg_0210.jpg", "neg_0212.jpg", "neg_0215.jpg", "neg_0216.jpg", "neg_0217.jpg",
    "neg_0218.jpg", "neg_0222.jpg", "neg_0223.jpg", "neg_0228.jpg", "neg_0230.jpg",
    "neg_0239.jpg", "neg_0240.jpg", "neg_0243.jpg", "neg_0247.jpg", "neg_0248.jpg",
    "neg_0250.jpg",
}

# kalite kontrolde bulunan manuel duzeltmeler (processed_data adi -> kutular).
# Ozel karar 2: Mustang ati / Dark Horse, F-150, Raptor, ST, FOCUS gibi model
# rozetleri Ford logosu sayilmaz. Bu gorsellerde kutular yalnizca bu rozetlere
# cizildigi icin bos birakilir (gorsel zor negatife gecer).
# ford_0181: yazisiz bos oval (ozel karar 4). ford_0308 (FORD yazisi) ve
# ford_0260, ford_0313, ford_0314, ford_0328 (kosedeki kucuk Ford ovali)
# kutulari dogru oldugu icin listede yok.
LABEL_OVERRIDES = {
    "ford_0082.jpg": [],
    "ford_0147.jpg": [],
    "ford_0156.jpg": [],
    "ford_0158.jpg": [],
    "ford_0161.jpg": [],
    "ford_0164.jpg": [],
    "ford_0168.jpg": [],
    "ford_0175.jpg": [],
    "ford_0181.jpg": [],
    "ford_0182.jpg": [],
    "ford_0228.jpg": [],
    "ford_0233.jpg": [],
    "ford_0240.jpg": [],
    "ford_0250.jpg": [],
    "ford_0253.jpg": [],
    "ford_0266.jpg": [],
    "ford_0272.jpg": [],
    "ford_0276.jpg": [],
    "ford_0280.jpg": [],
    "ford_0281.jpg": [],
    "ford_0282.jpg": [],
    "ford_0283.jpg": [],
    "ford_0284.jpg": [],
    "ford_0285.jpg": [],
    "ford_0286.jpg": [],
    "ford_0287.jpg": [],
    "ford_0288.jpg": [],
    "ford_0289.jpg": [],
    "ford_0290.jpg": [],
    "ford_0291.jpg": [],
    "ford_0292.jpg": [],
    "ford_0293.jpg": [],
    "ford_0294.jpg": [],
    "ford_0295.jpg": [],
    "ford_0296.jpg": [],
    "ford_0297.jpg": [],
    "ford_0298.jpg": [],
    "ford_0299.jpg": [],
    "ford_0300.jpg": [],
    "ford_0301.jpg": [],
    "ford_0302.jpg": [],
    "ford_0303.jpg": [],
    "ford_0304.jpg": [],
    "ford_0305.jpg": [],
    "ford_0306.jpg": [],
    "ford_0307.jpg": [],
    "ford_0309.jpg": [],
    "ford_0310.jpg": [],
    "ford_0311.jpg": [],
    "ford_0312.jpg": [],
    "ford_0315.jpg": [],
    "ford_0316.jpg": [],
    "ford_0317.jpg": [],
    "ford_0318.jpg": [],
    "ford_0319.jpg": [],
    "ford_0320.jpg": [],
    "ford_0321.jpg": [],
    "ford_0322.jpg": [],
    "ford_0323.jpg": [],
    "ford_0324.jpg": [],
    "ford_0325.jpg": [],
    "ford_0326.jpg": [],
    "ford_0327.jpg": [],
    "ford_0329.jpg": [],
    "ford_0330.jpg": [],
    "ford_0331.jpg": [],
    "ford_0332.jpg": [],
    "ford_0333.jpg": [],
    "ford_0334.jpg": [],
    "ford_0335.jpg": [],
    "ford_0336.jpg": [],
    "ford_0337.jpg": [],
    "ford_0338.jpg": [],
    "ford_0339.jpg": [],
    "ford_0340.jpg": [],
    "ford_0359.jpg": [],
    "ford_0364.jpg": [],
    "ford_0366.jpg": [],
    # son ad ford_0252: eski arma, ozel karar 3 geregi yalnizca FORD yazisi kutulanir
    # (%10 yeniden etiketlemede bulundu; ilk etiket tum armayi kapsiyordu)
    "ford_0345.jpg": [(0.490723, 0.259410, 0.409180, 0.132248)],
    # son ad ford_0249: fordlogo7'de ayni yaziya ic ice iki kutu var (krom cerceve + yazi);
    # sik olan yazi kutusu tutulur
    "ford_0342.jpg": [(0.490762, 0.286691, 0.540416, 0.110072)],
}

HASH_MAX_DIST = 4       # preprocess.py ile ayni esik
ASPECT_TOL = 0.02       # en-boy orani farki toleransi
MIN_BOX_PX = 10         # odev kurali: kisa kenari ~10px'den kucuk logolar etiketlenmez

NEG_NOTES = {
    "Subaru": "zor negatif - benzer oval logo (Subaru)",
    "Hyundai": "zor negatif - baska marka oval amblem (Hyundai)",
    "Toyota": "zor negatif - baska marka oval amblem (Toyota)",
    "Kia": "zor negatif - baska marka amblem (Kia)",
    "debadged": "zor negatif - logosuz arac on yuzu",
    "blue oval": "zor negatif - benzer mavi/oval logo",
}


def load_processed():
    meta = {}
    with open(os.path.join(PROCESSED, "pre_metadata.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["file_name"]] = row
    items = {}
    for folder in ("positive", "negative"):
        for name in sorted(os.listdir(os.path.join(PROCESSED, folder))):
            if not name.endswith(".jpg"):
                continue
            path = os.path.join(PROCESSED, folder, name)
            with Image.open(path) as im:
                items[name] = {
                    "path": path, "folder": folder, "size": im.size,
                    "phash": imagehash.phash(im),
                    "url": meta.get(name, {}).get("source_url", "unknown"),
                    "query": meta.get(name, {}).get("notes", ""),
                }
    return items


def read_yolo(label_path, w, h, warn_prefix):
    boxes = []
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        for line in f.read().split("\n"):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"  ! {warn_prefix}: hatali satir atlandi -> {line!r}")
                continue
            xc, yc, bw, bh = map(float, parts[1:])
            # kutuyu [0,1] icine kirp (Roboflow bazen 1.0000001 uretir)
            x1, y1 = max(0.0, xc - bw / 2), max(0.0, yc - bh / 2)
            x2, y2 = min(1.0, xc + bw / 2), min(1.0, yc + bh / 2)
            if x2 <= x1 or y2 <= y1:
                continue
            if min((x2 - x1) * w, (y2 - y1) * h) < MIN_BOX_PX:
                print(f"  ! {warn_prefix}: {MIN_BOX_PX}px'den kucuk kutu (kontrol edin)")
            boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def drop_nested(boxes, tag):
    """Kural: oval gorunuyorsa kutu ovalin dis sinirini kapsar. Ovalin icine ayrica
    cizilmis "Ford" yazisi kutularini (baska bir kutunun %95+ icinde, alani %15-60) siler."""
    rects = [(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y, w, h in boxes]
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    keep = []
    for i, a in enumerate(rects):
        nested = False
        for j, c in enumerate(rects):
            if i == j:
                continue
            inter = max(0, min(a[2], c[2]) - max(a[0], c[0])) * max(0, min(a[3], c[3]) - max(a[1], c[1]))
            if inter / area(a) > 0.95 and 0.15 < area(a) / area(c) < 0.6:
                nested = True
                break
        if nested:
            print(f"  - {tag}: oval icindeki yazi kutusu silindi")
        else:
            keep.append(boxes[i])
    return keep


def collect_labels(items):
    """processed adi -> kutu listesi. Sonraki export oncekini ezer."""
    hashes = [(n, it["phash"]) for n, it in items.items()]
    labels = {}
    for export in EXPORTS:
        found = {}
        for split in ("train", "valid", "test"):
            img_dir = os.path.join(export, split, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                with Image.open(os.path.join(img_dir, fname)) as im:
                    h = imagehash.phash(im)
                    size = im.size
                name, dist = min(((n, h - ph) for n, ph in hashes), key=lambda t: t[1])
                tag = f"{os.path.basename(export)}/{split}/{fname[:30]}"
                if dist > HASH_MAX_DIST:
                    # preprocess.py'nin eledigi gorsel (kucuk / tekrar)
                    continue
                pw, ph_ = items[name]["size"]
                if abs(size[0] / size[1] - pw / ph_) > ASPECT_TOL * (pw / ph_):
                    print(f"  ! {tag}: {name} ile en-boy orani farkli, atlandi")
                    continue
                lbl = os.path.join(export, split, "labels", os.path.splitext(fname)[0] + ".txt")
                boxes = read_yolo(lbl, pw, ph_, f"{tag} -> {name}")
                if boxes is None:
                    continue
                boxes = drop_nested(boxes, f"{tag} -> {name}")
                exact = size == (pw, ph_)
                prev = found.get(name)
                # ayni gorselin birden fazla kopyasi varsa birebir boyutlu olani tercih et
                if prev is None or (exact and not prev[1]):
                    found[name] = (boxes, exact)
        print(f"[{os.path.basename(export)}] eslesen gorsel: {len(found)}")
        labels.update({n: b for n, (b, _) in found.items()})
    labels.update(LABEL_OVERRIDES)
    return labels


def note_for(item, has_logo):
    q = item["query"]
    if item["folder"] == "negative" and has_logo:
        return f"negatif sorgusunda cikan Ford logosu - logo izgarasi/tabela ({q})"
    if item["folder"] == "negative":
        for key, note in NEG_NOTES.items():
            if key.lower() in q.lower():
                return note
        return "zor negatif"
    if not has_logo:
        return f"zor negatif - Ford logosu yok, model rozeti ({q})"
    return q


def main():
    items = load_processed()
    labels = collect_labels(items)

    for d in (POOL, TO_LABEL):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(POOL, "images"))
    os.makedirs(os.path.join(POOL, "labels"))
    os.makedirs(TO_LABEL)

    pos, neg, pending = [], [], []
    for name, it in items.items():
        if name in labels:
            (pos if labels[name] else neg).append(name)
        elif it["folder"] == "negative" and name not in NEG_NEEDS_REVIEW:
            labels[name] = []
            neg.append(name)
        else:
            pending.append(name)

    # etiketleme turu sirasiyla adlandir ki yeni tur eski adlari kaydirmasin:
    # orijinal pozitifler, negatiften pozitife gecenler, kucuk logo turu
    def round_of(n):
        if items[n]["folder"] == "negative":
            return 1
        return 2 if int(n[5:9]) >= SMALL_LOGO_START else 0
    pos.sort(key=lambda n: (round_of(n), n))
    # orijinal negatifler once, sonra pozitiften negatife gecenler
    neg.sort(key=lambda n: (items[n]["folder"] != "negative", round_of(n), n))
    rows = []
    for prefix, group in (("ford", pos), ("neg", neg)):
        for i, name in enumerate(group, 1):
            it = items[name]
            new = f"{prefix}_{i:04d}"
            shutil.copy2(it["path"], os.path.join(POOL, "images", new + ".jpg"))
            with open(os.path.join(POOL, "labels", new + ".txt"), "w") as f:
                f.writelines(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in labels[name])
            rows.append({
                "file_name": new + ".jpg", "has_logo": int(prefix == "ford"),
                "num_boxes": len(labels[name]), "width": it["size"][0],
                "height": it["size"][1], "source_url": it["url"],
                "notes": note_for(it, prefix == "ford"),
                "orig_name": name, "phash": str(it["phash"]),
            })

    with open(os.path.join(POOL, "pool.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for name in sorted(pending):
        shutil.copy2(items[name]["path"], os.path.join(TO_LABEL, name))

    print(f"havuz -> pozitif: {len(pos)} | negatif: {len(neg)} | kutu: {sum(r['num_boxes'] for r in rows)}")
    print(f"etiketlenmeyi bekleyen: {len(pending)} -> {TO_LABEL}")


if __name__ == "__main__":
    main()
