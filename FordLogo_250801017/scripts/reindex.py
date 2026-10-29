import os
import csv

def reindex_and_clean():
    meta_path = "processed_data/pre_metadata.csv"
    if not os.path.exists(meta_path):
        print("error: pre_metadata.csv not found")
        return

    with open(meta_path, mode="r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    new_records = []
    pos_count = 1
    neg_count = 1

    # create folder for negative empty labels
    neg_label_dir = "processed_data/negative_labels"
    os.makedirs(neg_label_dir, exist_ok=True)

    # first rename to temp names to avoid overwrite collisions
    temp_moves = []
    for row in records:
        old_name = row["file_name"]
        folder = "positive" if int(row["has_logo"]) == 1 else "negative"
        old_path = os.path.join("processed_data", folder, old_name)

        # skip if user manually deleted the image
        if not os.path.exists(old_path):
            continue

        temp_path = os.path.join("processed_data", folder, f"temp_{old_name}")
        os.rename(old_path, temp_path)
        temp_moves.append((temp_path, folder, row))

    # rename to sequential standard names
    for temp_path, folder, row in temp_moves:
        if folder == "positive":
            new_name = f"ford_{pos_count:04d}.jpg"
            pos_count += 1
        else:
            new_name = f"neg_{neg_count:04d}.jpg"
            neg_count += 1
            # create empty .txt file for negative image
            txt_name = new_name.replace(".jpg", ".txt")
            open(os.path.join(neg_label_dir, txt_name), "w").close()

        new_path = os.path.join("processed_data", folder, new_name)
        os.rename(temp_path, new_path)

        row["file_name"] = new_name
        new_records.append(row)

    # update metadata csv
    with open(meta_path, mode="w", newline="", encoding="utf-8") as f:
        fields = ["file_name", "has_logo", "width", "height", "source_url", "notes"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(new_records)

    print(f"reindexed -> positive: {pos_count - 1} | negative: {neg_count - 1}")
    print("empty .txt files generated for negatives.")

if __name__ == "__main__":
    reindex_and_clean()