"""
Kucuk / uzak Ford logolari icin ek gorsel toplama (ikinci tur).

Ilk turda kisa kenari < 32 px olan logo sayisi az oldugu icin genis acili sahne
gorselleri (cadde, otopark, bayi binasi, trafik) aranir. Indirilen gorseller
preprocess.py ile ayni kurallarla (EXIF, RGB/JPG, >=320 px, phash <= 4 tekrar
kontrolu) islenir; mevcut tum gorsellerle tekrar kontrolu yapilir ve numaralandirma
mevcut son ford_XXXX numarasindan devam eder.

Cikti: raw_data/small_logo/ (ham), processed_data/positive/ford_XXXX.jpg,
       processed_data/pre_metadata.csv (yeni satirlar eklenir)
"""
import os
import sys
import csv
from PIL import Image, ImageOps
import imagehash
from download_images import scrape_images

RAW = "raw_data/small_logo"
OUT = "processed_data/positive"
META = "processed_data/pre_metadata.csv"
MIN_DIM = 320

QUERIES = [
    "Ford dealership exterior building",
    "Ford cars parked on street",
    "Ford F-150 parked side view",
    "Ford Transit van on street",
    "Ford Ranger driving on road",
    "Ford Explorer in parking lot",
    "Ford Focus in city traffic",
    "Ford Kuga on the road",
    "Ford car show crowd",
    "Ford police interceptor on street",
]


def existing_hashes():
    hashes = []
    for sub in ("positive", "negative"):
        d = os.path.join("processed_data", sub)
        for f in os.listdir(d):
            if f.endswith(".jpg"):
                with Image.open(os.path.join(d, f)) as im:
                    hashes.append(imagehash.phash(im))
    return hashes


def process():
    url_map = {}
    with open(os.path.join(RAW, "raw_urls.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url_map[os.path.join(row["group"], row["temp_filename"])] = (row["source_url"], row["query"])

    seen = existing_hashes()
    nums = [int(f[5:9]) for f in os.listdir(OUT) if f.startswith("ford_")]
    count = max(nums) + 1
    records, small, dup, err = [], 0, 0, 0
    for sub in sorted(os.listdir(RAW)):
        sub_path = os.path.join(RAW, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            try:
                with Image.open(os.path.join(sub_path, fname)) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size
                    if min(w, h) < MIN_DIM:
                        small += 1
                        continue
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    ph = imagehash.phash(img)
                    if any(ph - s <= 4 for s in seen):
                        dup += 1
                        continue
                    seen.append(ph)
                    name = f"ford_{count:04d}.jpg"
                    img.save(os.path.join(OUT, name), "JPEG", quality=95)
            except Exception:
                err += 1
                continue
            url, query = url_map.get(os.path.join(sub, fname), ("unknown", ""))
            records.append({"file_name": name, "has_logo": 1, "width": w, "height": h,
                            "source_url": url, "notes": query})
            count += 1

    with open(META, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["file_name", "has_logo", "width", "height",
                                      "source_url", "notes"]).writerows(records)
    print(f"[small_logo] kaydedilen: {len(records)} | kucuk: {small} | tekrar: {dup} | hata: {err}")


if __name__ == "__main__":
    # 1) python download_small_logos.py download
    # 2) raw_data/small_logo icinde logosu buyuk / Ford olmayan gorseller elle silinir
    # 3) python download_small_logos.py process
    if sys.argv[1:] == ["download"]:
        scrape_images(QUERIES, RAW, "small", max_num=30)
    elif sys.argv[1:] == ["process"]:
        process()
    else:
        print(__doc__)
