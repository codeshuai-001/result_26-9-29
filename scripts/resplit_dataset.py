#!/usr/bin/env python3
"""Stratified 80/20 resplit of the TLFO dataset so every class appears in train and val."""
from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("/root/AKCMamba_YOLO/datasets/tlfo")
DST = Path("/root/AKCMamba_YOLO/datasets/tlfo_balanced")
VAL_RATIO = 0.2
SEED = 42
NAMES = {0: "nest", 1: "kite", 2: "balloon", 3: "trash"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def primary_class(label_path: Path) -> int:
    classes = []
    text = label_path.read_text(errors="replace").strip()
    if not text:
        return -1
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        classes.append(int(float(parts[0])))
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]


def collect():
    items = []
    for split in ("train", "val"):
        img_dir = SRC / "images" / split
        lab_dir = SRC / "labels" / split
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                raise FileNotFoundError(lab)
            items.append((img, lab, primary_class(lab)))
    return items


def split_items(items):
    rng = random.Random(SEED)
    by_cls = defaultdict(list)
    for item in items:
        by_cls[item[2]].append(item)
    train, val = [], []
    print("=== source class image counts ===")
    for cid in sorted(by_cls):
        group = by_cls[cid]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * VAL_RATIO)))
        if n_val >= len(group):
            n_val = max(1, len(group) // 5)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
        print(f"  {NAMES.get(cid, cid)}: total={len(group)} train={len(group)-n_val} val={n_val}")
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def reset_dst():
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(pairs, split: str):
    cls_inst = Counter()
    for img, lab, _cid in pairs:
        link_or_copy(img, DST / "images" / split / img.name)
        link_or_copy(lab, DST / "labels" / split / lab.name)
        for line in lab.read_text(errors="replace").splitlines():
            parts = line.split()
            if parts:
                cls_inst[int(float(parts[0]))] += 1
    print(
        f"=== {split} images={len(pairs)} instances="
        f"{dict((NAMES.get(k, k), v) for k, v in sorted(cls_inst.items()))}"
    )


def main():
    items = collect()
    print(f"collected {len(items)} image/label pairs")
    train, val = split_items(items)
    overlap = {p[0].stem for p in train} & {p[0].stem for p in val}
    if overlap:
        raise RuntimeError(f"split leakage: {len(overlap)}")
    reset_dst()
    materialize(train, "train")
    materialize(val, "val")
    print(f"done -> {DST}")


if __name__ == "__main__":
    main()
