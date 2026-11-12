import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Folder containing one subfolder per class")
    parser.add_argument("--output", default=Path("data/processed"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train", default=0.70, type=float)
    parser.add_argument("--val", default=0.15, type=float)
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        raise SystemExit("Train and validation fractions must be positive and sum to less than 1.")

    rng = random.Random(args.seed)
    class_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit("No class subdirectories found.")

    total = 0
    for class_dir in class_dirs:
        images = sorted(p for p in class_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 3:
            print(f"Skipping {class_dir.name}: needs at least 3 images, found {len(images)}")
            continue
        rng.shuffle(images)
        n = len(images)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1

        partitions = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split, paths in partitions.items():
            destination = args.output / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(paths):
                shutil.copy2(source, destination / f"{index:06d}_{source.name}")
        print(class_dir.name + ": " + ", ".join(
            f"{key}={len(value)}" for key, value in partitions.items()))
        total += n

    print(f"Copied {total} images into {args.output.resolve()}")


if __name__ == "__main__":
    main()
