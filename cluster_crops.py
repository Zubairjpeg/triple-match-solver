"""
Cluster crops by visual similarity. For each cluster, picks the medoid
(crop most similar to all others in its cluster) as the representative.

Outputs:
  - staging/_clusters.png  : grid showing each cluster's representative + count
  - staging/clusters/<id>.png : representative crop per cluster
"""
import shutil
from pathlib import Path
import cv2
import numpy as np


SIM_THRESHOLD = 0.78  # 0..1, higher = stricter
RESIZE = 80


def feature(img: np.ndarray) -> np.ndarray:
    """Flat grayscale feature vector."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32).flatten() / 255.0


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    # normalized cross-correlation
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom == 0:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def main():
    staging = Path("staging")
    paths = sorted(p for p in staging.glob("*.png") if not p.name.startswith("_") and "/clusters" not in str(p))
    if not paths:
        print("No crops to cluster")
        return

    feats = [feature(cv2.imread(str(p))) for p in paths]

    # Greedy clustering: for each crop, place in first cluster with similarity > threshold
    clusters: list[list[int]] = []
    for i, f in enumerate(feats):
        placed = False
        for cluster in clusters:
            # similarity to cluster representative (first member)
            sim = similarity(f, feats[cluster[0]])
            if sim >= SIM_THRESHOLD:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # Output
    out_dir = staging / "clusters"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Found {len(clusters)} clusters across {len(paths)} crops")
    print()

    # Per-cluster: pick medoid (highest avg similarity)
    for cid, members in enumerate(clusters):
        if len(members) == 1:
            medoid_idx = members[0]
        else:
            best, best_score = members[0], -1.0
            for m in members:
                avg = sum(similarity(feats[m], feats[n]) for n in members if n != m) / max(1, len(members) - 1)
                if avg > best_score:
                    best_score = avg
                    best = m
            medoid_idx = best
        rep = cv2.imread(str(paths[medoid_idx]))
        cv2.imwrite(str(out_dir / f"cluster_{cid:02d}.png"), rep)
        member_names = ", ".join(paths[m].name.split("_")[1] + paths[m].name.split("_")[2] for m in members)
        print(f"cluster {cid:02d} ({len(members)} members, medoid={paths[medoid_idx].name})")

    # Grid: one cell per cluster, label with count + id
    tile = 200
    cols = 5
    rows = (len(clusters) + cols - 1) // cols
    cell_w, cell_h = tile + 16, tile + 40
    mosaic = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)
    for cid, members in enumerate(clusters):
        if len(members) == 1:
            medoid_idx = members[0]
        else:
            best, best_score = members[0], -1.0
            for m in members:
                avg = sum(similarity(feats[m], feats[n]) for n in members if n != m) / max(1, len(members) - 1)
                if avg > best_score:
                    best_score = avg
                    best = m
            medoid_idx = best
        img = cv2.imread(str(paths[medoid_idx]))
        img = cv2.resize(img, (tile, tile))
        r, c = divmod(cid, cols)
        y, x = r * cell_h + 6, c * cell_w + 8
        mosaic[y:y + tile, x:x + tile] = img
        label = f"#{cid:02d}  n={len(members)}"
        cv2.putText(mosaic, label, (x + 4, y + tile + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(staging / "_clusters.png"), mosaic)
    print(f"\nWrote staging/_clusters.png and {len(clusters)} representative images")


if __name__ == "__main__":
    main()
