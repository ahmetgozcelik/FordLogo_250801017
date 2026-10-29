"""
Rapor figurlerini veri setinden uretir -> figures/
Calistirma: python make_figures.py
"""
import os
import csv
import glob
import shutil
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "FordLogo_250801017")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

META = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(DS, "metadata.csv"), encoding="utf-8"))}
try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
except OSError:
    FONT = ImageFont.load_default()


def load(name):
    s = META[name]["split"]
    im = Image.open(os.path.join(DS, "images", s, name)).convert("RGB")
    boxes = []
    for l in open(os.path.join(DS, "labels", s, name[:-4] + ".txt")).read().splitlines():
        if l.strip():
            _, x, y, w, h = map(float, l.split())
            boxes.append((x, y, w, h))
    return im, boxes


def draw(im, boxes, color="red", width=None):
    W, H = im.size
    width = width or max(3, W // 250)
    d = ImageDraw.Draw(im)
    for x, y, w, h in boxes:
        d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                    outline=color, width=width)
    return im


def tile(im, size):
    """oran korunarak size kutusuna sigdirir, beyaz zemine ortalar"""
    im = im.copy()
    im.thumbnail(size)
    bg = Image.new("RGB", size, "white")
    bg.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return bg


def grid(cells, cols, size, captions=None, path=None):
    rows = (len(cells) + cols - 1) // cols
    cap = 40 if captions else 0
    sheet = Image.new("RGB", (cols * size[0], rows * (size[1] + cap)), "white")
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(cells):
        x, y = (i % cols) * size[0], (i // cols) * (size[1] + cap)
        sheet.paste(tile(c, (size[0] - 8, size[1] - 8)), (x + 4, y + 4))
        if captions:
            d.text((x + 8, y + size[1] + 4), captions[i], fill="black", font=FONT)
    sheet.save(path, quality=90)
    print("->", os.path.relpath(path, HERE))


def first_with(note, split="train", min_boxes=1):
    return next(n for n, r in sorted(META.items())
                if r["notes"] == note and r["split"] == split and int(r["num_boxes"]) >= min_boxes)


# 1) farkli baglamlardan etiketli ornekler
ctx = [("Ford car front grille emblem", "ön ızgara"),
       ("Ford car rear tailgate badge", "bagaj kapağı"),
       ("Ford dealership sign night", "bayi tabelası (gece)"),
       ("Ford steering wheel logo closeup", "direksiyon"),
       ("Ford key fob logo", "anahtarlık"),
       ("vintage Ford car badge", "eski rozet"),
       ("Ford dealership exterior building", "bayi binası (küçük)"),
       ("Ford car show crowd", "fuar (küçük)")]
names = [first_with(q) for q, _ in ctx]
grid([draw(*load(n), width=max(4, load(n)[0].width // 90)) for n in names], 4, (480, 360),
     [f"{c} - {n[:-4]}" for (_, c), n in zip(ctx, names)], os.path.join(OUT, "ornekler.jpg"))

# 2) zor negatif ornekleri (her kategoriden bir tane)
cats = [("zor negatif - benzer oval logo (Subaru)", "Subaru ovali"),
        ("zor negatif - baska marka oval amblem (Hyundai)", "Hyundai"),
        ("zor negatif - baska marka oval amblem (Toyota)", "Toyota"),
        ("zor negatif - baska marka amblem (Kia)", "Kia"),
        ("zor negatif - logosuz arac on yuzu", "rozetsiz ızgara"),
        ("zor negatif - benzer mavi/oval logo", "mavi oval logo")]
names = [next(n for n, r in sorted(META.items()) if r["notes"] == q) for q, _ in cats]
names.append(next(n for n, r in sorted(META.items()) if r["has_logo"] == "0" and "model rozeti" in r["notes"]
                  and "Mustang" in r["notes"]))
names.append(next(n for n, r in sorted(META.items()) if r["has_logo"] == "0" and "F-150" in r["notes"]))
caps = [c for _, c in cats] + ["Mustang atı", "F-150 rozeti"]
grid([load(n)[0] for n in names], 4, (480, 360), [f"{c} - {n[:-4]}" for c, n in zip(caps, names)],
     os.path.join(OUT, "zor_negatifler.jpg"))

# 3) kucuk logo ornekleri: tam gorsel + kutu cevresinden buyutulmus kesit
small = []
for n, r in sorted(META.items()):
    if r["has_logo"] != "1":
        continue
    im, boxes = load(n)
    W, H = im.size
    if len(boxes) == 1 and min(boxes[0][2] * W, boxes[0][3] * H) < 20 and int(n[5:9]) >= 319:
        small.append(n)
cells, caps = [], []
for n in small[:3]:
    im, boxes = load(n)
    W, H = im.size
    x, y, w, h = boxes[0]
    side = max(w * W, h * H) * 4
    crop = im.crop((x * W - side / 2, y * H - side / 2, x * W + side / 2, y * H + side / 2))
    bw, bh = w * W * crop.width / side, h * H * crop.height / side
    full = draw(im.copy(), boxes, width=max(4, W // 150))
    crop = crop.resize((360, 360), Image.LANCZOS)
    s = 360 / side
    ImageDraw.Draw(crop).rectangle([180 - w * W * s / 2, 180 - h * H * s / 2,
                                    180 + w * W * s / 2, 180 + h * H * s / 2], outline="red", width=3)
    cells += [full, crop]
    caps += [f"{n[:-4]} ({W}x{H})", f"kesit: kutu {w * W:.0f}x{h * H:.0f} px"]
grid(cells, 2, (540, 380), caps, os.path.join(OUT, "kucuk_logolar.jpg"))

# 4) yeniden etiketleme tutarsizliklari (yesil: ilk, kirmizi: ikinci etiketleme)
cells, caps = [], []
for n, export, first in (("ford_0252.jpg", "fordlogo2", [(0.491211, 0.489827, 0.933594, 0.871821)]),
                         ("ford_0366.jpg", "fordlogo5", None)):
    im, boxes = load(n)
    first = first or boxes
    lbl = glob.glob(os.path.join(HERE, "..", export, "*", "labels", n[:-4] + "_jpg*.txt"))[0]
    second = [tuple(map(float, l.split()[1:])) for l in open(lbl).read().splitlines() if l.strip()]
    im = draw(im, first, "lime", max(6, im.width // 120))
    im = draw(im, second, "red", max(3, im.width // 300))
    if n == "ford_0366.jpg":  # kutular ust seritte kucuk kaliyor, o bolgeyi buyut
        im = im.crop((int(im.width * 0.3), 0, int(im.width * 0.9), int(im.height * 0.3)))
    cells.append(im)
    caps.append(n[:-4])
grid(cells, 2, (640, 520), caps, os.path.join(OUT, "tutarsizliklar.jpg"))

# 5) letterbox: panodan ilk 3 satir
pano = Image.open(os.path.join(DS, "letterbox_ornekleri", "_pano.jpg"))
pano.crop((0, 0, pano.width, pano.height // 2)).save(os.path.join(OUT, "letterbox.jpg"), quality=90)

# 6) augmentation panosu: hucreler arasindaki bos beyaz satirlari kaldir
aug = Image.open(os.path.join(DS, "augmentation_ornekleri", "aug_ford_0042.jpg")).convert("L")
px = aug.load()
keep = [y for y in range(aug.height) if any(px[x, y] < 245 for x in range(0, aug.width, 3))]
rows, prev = [], None
for y in keep:
    if prev is not None and y - prev > 12:
        rows.append(None)
    rows.append(y)
    prev = y
src = Image.open(os.path.join(DS, "augmentation_ornekleri", "aug_ford_0042.jpg")).convert("RGB")
out = Image.new("RGB", (src.width, len(rows)), "white")
for i, y in enumerate(rows):
    if y is not None:
        out.paste(src.crop((0, y, src.width, y + 1)), (0, i))
out.save(os.path.join(OUT, "augmentation.jpg"), quality=90)

# 7) hazir gorseller
for src, dst in (
                 ("kalite_kontrol/_pano.jpg", "kalite_kontrol.jpg"),
                 ("istatistikler/pozitif_negatif.png", "pozitif_negatif.png"),
                 ("istatistikler/kutu_sayisi_histogram.png", "kutu_sayisi.png"),
                 ("istatistikler/kutu_boyutu_dagilimi.png", "kutu_boyutu.png"),
                 ("istatistikler/cozunurluk_dagilimi.png", "cozunurluk.png")):
    shutil.copy2(os.path.join(DS, src), os.path.join(OUT, dst))
print("bitti")
