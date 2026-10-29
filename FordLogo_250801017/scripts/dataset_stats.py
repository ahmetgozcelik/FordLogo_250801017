"""
Veri seti istatistikleri ve rapor grafikleri -> istatistikler/
- bolum bazinda gorsel / pozitif / negatif / kutu sayilari (tablo: istatistikler.md)
- gorsel basina kutu sayisi histogrami
- kutu boyutu dagilimi (kutu alani / gorsel alani)
- gorsel cozunurluk dagilimi
- kanal bazinda mean / std (yalnizca TRAIN uzerinden; val/test sizintisi olmasin)
README.md icindeki <!-- STATS --> ... <!-- /STATS --> blogunu otomatik gunceller.
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from common import DATASET, SPLITS, list_images, image_path, read_boxes

OUT = os.path.join(DATASET, "istatistikler")
os.makedirs(OUT, exist_ok=True)

table, boxes_per_img, area_ratio, resolutions, short_px = [], [], [], [], []
for s in SPLITS:
    names = list_images(s)
    pos = [n for n in names if n.startswith("ford_")]
    nbox = 0
    for n in names:
        b = read_boxes(s, n)
        nbox += len(b)
        with Image.open(image_path(s, n)) as im:
            W, H = im.size
        resolutions.append((W, H))
        if b:
            boxes_per_img.append(len(b))
        for _, _, _, w, h in b:
            area_ratio.append(w * h)
            short_px.append(min(w * W, h * H))
    table.append((s, len(names), len(pos), len(names) - len(pos), nbox))

# kanal ortalama / std (train, piksel bazinda, 0-1 olcek)
sums, sq, count = np.zeros(3), np.zeros(3), 0
for n in list_images("train"):
    a = np.asarray(Image.open(image_path("train", n)).convert("RGB"), dtype=np.float64) / 255.0
    a = a.reshape(-1, 3)
    sums += a.sum(0)
    sq += (a ** 2).sum(0)
    count += a.shape[0]
mean = sums / count
std = np.sqrt(sq / count - mean ** 2)

# grafikler
plt.figure(figsize=(5, 3.5))
plt.hist(boxes_per_img, bins=range(1, max(boxes_per_img) + 2), align="left", color="#3b6ea5", edgecolor="white")
plt.xlabel("Görsel başına kutu sayısı"); plt.ylabel("Pozitif görsel sayısı")
plt.title("Görsel başına kutu sayısı"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "kutu_sayisi_histogram.png"), dpi=150); plt.close()

plt.figure(figsize=(5, 3.5))
plt.hist(np.array(area_ratio) * 100, bins=np.logspace(-2, 2, 30), color="#3b6ea5", edgecolor="white")
plt.xscale("log"); plt.xlabel("Kutu alanı / görsel alanı (%)"); plt.ylabel("Kutu sayısı")
plt.title("Kutu boyutu dağılımı"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "kutu_boyutu_dagilimi.png"), dpi=150); plt.close()

r = np.array(resolutions)
plt.figure(figsize=(5, 4))
plt.scatter(r[:, 0], r[:, 1], s=8, alpha=0.5, color="#3b6ea5")
plt.xlabel("Genişlik (px)"); plt.ylabel("Yükseklik (px)"); plt.title("Görsel çözünürlük dağılımı")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cozunurluk_dagilimi.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(len(SPLITS))
ax.bar(x - 0.2, [t[2] for t in table], 0.4, label="pozitif", color="#3b6ea5")
ax.bar(x + 0.2, [t[3] for t in table], 0.4, label="negatif", color="#c9824a")
ax.set_xticks(x, SPLITS); ax.set_ylabel("Görsel sayısı"); ax.legend(); ax.set_title("Pozitif / negatif dağılımı")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "pozitif_negatif.png"), dpi=150); plt.close()

tot = [sum(t[i] for t in table) for i in range(1, 5)]
lines = ["| Bölüm | Görsel | Pozitif | Negatif | Kutu |", "|---|---:|---:|---:|---:|"]
lines += [f"| {s} | {n} | {p} | {g} | {b} |" for s, n, p, g, b in table]
lines += [f"| **Toplam** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{tot[3]}** |", ""]
lines += [f"- Görsel başına kutu (pozitiflerde): ort. {np.mean(boxes_per_img):.2f}, maks. {max(boxes_per_img)}",
          f"- Kutu alanı / görsel alanı: medyan %{np.median(area_ratio) * 100:.2f}, "
          f"min %{min(area_ratio) * 100:.3f}, maks %{max(area_ratio) * 100:.1f}",
          f"- Kutu kısa kenarı < 32 px olan küçük logo sayısı: {sum(v < 32 for v in short_px)}",
          f"- Çözünürlük: genişlik {r[:, 0].min()}-{r[:, 0].max()} px, yükseklik {r[:, 1].min()}-{r[:, 1].max()} px, "
          f"medyan {int(np.median(r[:, 0]))}x{int(np.median(r[:, 1]))}",
          "",
          "Kanal bazında ortalama / std (RGB, 0–1 ölçek, yalnızca train kümesi, "
          f"{count:,} piksel):",
          "",
          "```python",
          f"mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]",
          f"std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]",
          "```"]
md = "\n".join(lines)
open(os.path.join(OUT, "istatistikler.md"), "w").write(md + "\n")
print(md)

readme = os.path.join(DATASET, "README.md")
if os.path.exists(readme):
    txt = open(readme, encoding="utf-8").read()
    txt = re.sub(r"<!-- STATS -->.*<!-- /STATS -->", lambda _: f"<!-- STATS -->\n{md}\n<!-- /STATS -->", txt, flags=re.S)
    open(readme, "w", encoding="utf-8").write(txt)
