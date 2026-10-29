"""
Etiketli havuzu (build_dataset.py ciktisi) %70 / %15 / %15 olarak train/val/test'e boler.

- Sabit tohum: SEED = 42
- Tabakali: pozitif ve negatifler ayri ayri bolunur, oranlar her bolumde korunur.
- Grup korumali: perceptual hash mesafesi <= GROUP_DIST olan (ayni foto serisinden /
  cok benzer) gorseller tek grup sayilir ve grubun tamami ayni bolume gider.

Cikti: FordLogo_<No>/images/{train,val,test}, labels/{train,val,test},
       metadata.csv, data.yaml, classes.txt

UYARI: Bu bolme sonraki odevlerde aynen kullanilacak; etiketleme bittikten sonra
bir kez calistirin ve sonra tekrar calistirmayin.
"""
import os
import csv
import random
import shutil
import imagehash

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_DIST = 10

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.abspath(os.path.join(HERE, ".."))
POOL = os.path.abspath(os.path.join(HERE, "..", "..", "FordLogo1", "Project", "labeled_pool"))
FIELDS = ["file_name", "split", "has_logo", "num_boxes", "width", "height", "source_url", "notes"]


def make_groups(rows):
    """union-find ile benzer gorselleri gruplar"""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [imagehash.hex_to_hash(r["phash"]) for r in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hashes[i] - hashes[j] <= GROUP_DIST:
                parent[find(i)] = find(j)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(find(i), []).append(r)
    return list(groups.values())


def split_class(groups, rng):
    """gruplari karistirip hedef sayilara gore doldurur"""
    rng.shuffle(groups)
    total = sum(len(g) for g in groups)
    targets = {s: round(total * p) for s, p in RATIOS.items()}
    targets["train"] = total - targets["val"] - targets["test"]
    counts = {s: 0 for s in RATIOS}
    out = {}
    # buyuk gruplar once yerlessin ki kucuk bolumlere tasma olmasin
    for g in sorted(groups, key=len, reverse=True):
        split = max(RATIOS, key=lambda s: (targets[s] - counts[s]) / targets[s])
        counts[split] += len(g)
        for r in g:
            out[r["file_name"]] = split
    return out


def main():
    with open(os.path.join(POOL, "pool.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = make_groups(rows)
    multi = [g for g in groups if len(g) > 1]
    print(f"{len(rows)} gorsel, {len(groups)} grup ({len(multi)} grup birden fazla gorsel iceriyor)")

    rng = random.Random(SEED)
    assign = {}
    for has_logo in ("1", "0"):
        cls_groups = [g for g in groups if g[0]["has_logo"] == has_logo]
        # karisik grup (pozitif + negatif) olursa cogunluga gore siniflandir
        assign.update(split_class(cls_groups, rng))
    mixed = [g for g in groups if len({r["has_logo"] for r in g}) > 1]
    for g in mixed:  # karisik gruplari tek bolume topla
        s = assign[g[0]["file_name"]]
        for r in g:
            assign[r["file_name"]] = s

    for sub in ("images", "labels"):
        for s in RATIOS:
            d = os.path.join(DATASET, sub, s)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)

    meta = []
    for r in sorted(rows, key=lambda r: (assign[r["file_name"]], r["file_name"])):
        s = assign[r["file_name"]]
        stem = os.path.splitext(r["file_name"])[0]
        shutil.copy2(os.path.join(POOL, "images", r["file_name"]), os.path.join(DATASET, "images", s))
        shutil.copy2(os.path.join(POOL, "labels", stem + ".txt"), os.path.join(DATASET, "labels", s))
        meta.append({k: (s if k == "split" else r[k]) for k in FIELDS})

    with open(os.path.join(DATASET, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(meta)

    with open(os.path.join(DATASET, "data.yaml"), "w") as f:
        f.write("path: .  # veri seti kok klasoru\n"
                "train: images/train\nval: images/val\ntest: images/test\n"
                "nc: 1\nnames: [\"ford_logo\"]\n")
    with open(os.path.join(DATASET, "classes.txt"), "w") as f:
        f.write("ford_logo\n")

    for s in RATIOS:
        part = [m for m in meta if m["split"] == s]
        pos = sum(m["has_logo"] == "1" for m in part)
        print(f"{s:5s}: {len(part):4d} gorsel | pozitif {pos} | negatif {len(part) - pos}")


if __name__ == "__main__":
    main()
