import os
import csv
from PIL import Image, ImageOps
import imagehash

def process_folder(raw_dir, out_dir, prefix, has_logo, seen_hashes, min_dim=320):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "raw_urls.csv")
    
    # load url mapping from scraper output
    url_map = {}
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = os.path.join(row["group"], row["temp_filename"])
                url_map[key] = (row["source_url"], row["query"])

    records = []
    count = 1
    skipped_size = 0
    skipped_dup = 0
    skipped_err = 0

    for sub_dir in sorted(os.listdir(raw_dir)):
        sub_path = os.path.join(raw_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
            
        for fname in sorted(os.listdir(sub_path)):
            fpath = os.path.join(sub_path, fname)
            if not os.path.isfile(fpath):
                continue
                
            try:
                with Image.open(fpath) as img:
                    # fix exif orientation before labeling
                    img = ImageOps.exif_transpose(img)
                    
                    # check min dimension (>= 320px)
                    w, h = img.size
                    if min(w, h) < min_dim:
                        skipped_size += 1
                        continue
                        
                    # convert to rgb
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        
                    # check duplicates with perceptual hash
                    p_hash = imagehash.phash(img)
                    is_dup = False
                    for existing_hash in seen_hashes:
                        if abs(p_hash - existing_hash) <= 4:
                            is_dup = True
                            break
                    if is_dup:
                        skipped_dup += 1
                        continue
                    seen_hashes.add(p_hash)
                    
                    # save as standardized jpg
                    new_name = f"{prefix}_{count:04d}.jpg"
                    out_path = os.path.join(out_dir, new_name)
                    img.save(out_path, "JPEG", quality=95)
                    
                    key = os.path.join(sub_dir, fname)
                    src_url, query_note = url_map.get(key, ("unknown", ""))
                    
                    records.append({
                        "file_name": new_name,
                        "has_logo": has_logo,
                        "width": w,
                        "height": h,
                        "source_url": src_url,
                        "notes": query_note
                    })
                    count += 1
                    
            except Exception:
                skipped_err += 1
                continue

    print(f"[{prefix}] saved: {count - 1} | small: {skipped_size} | dup: {skipped_dup} | err: {skipped_err}")
    return records

if __name__ == "__main__":
    hashes = set()
    all_records = []
    
    print("starting preprocessing...")
    
    pos_records = process_folder("raw_data/positive", "processed_data/positive", "ford", 1, hashes)
    all_records.extend(pos_records)
    
    neg_records = process_folder("raw_data/negative", "processed_data/negative", "neg", 0, hashes)
    all_records.extend(neg_records)
    
    # save temporary metadata before train/val/test split
    os.makedirs("processed_data", exist_ok=True)
    with open("processed_data/pre_metadata.csv", mode="w", newline="", encoding="utf-8") as f:
        fields = ["file_name", "has_logo", "width", "height", "source_url", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)
        
    print("done. files saved to processed_data/")