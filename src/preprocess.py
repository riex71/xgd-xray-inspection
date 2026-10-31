"""검사기 색 표시 제거(v3)와 흔적 무력화 학습 영상(v3tr).

원 protocol(4번 분석 모델 src/prepare.py · src/baselines.py · src/common.py)의 함수를 연산 그대로 옮겼다. 설정값만 config.yaml
preprocess 에서 읽는다. 만든 영상의 픽셀 해시를 기록(data/v3_pixel_sha256.json)과 대조해 한 장이라도 다르면 멈춘다.

  v3 (data/clean_v3/images, 435장 = 개발 356 + 평가 79)
      유채색 테두리(채도 > 40) 안쪽의 반투명 덮개(×0.80)를 표시별 이득으로 되돌리고, 테두리는 Telea 인페인팅 후 가장 덜
      겹치는 방향(±26px)의 고주파 질감을 옮겨 채운다. 검출기 입력(교차검증·평가셋·현장)은 모두 v3 이다.
  v3tr (data/v3tr_orig/images, 개발 356장 = DFL 학습 입력)
      v3 영상의 이물·실제 표시가 없는 제품 안쪽에 가짜 표시를 1~2개 그린 뒤 다시 v3 로 지운다(원본 해시로 정한 난수).
      표시 자리 = 이물 자리라는 지름길을 끊는다. 정답은 원본 TXT(data/labels_orig) 그대로.

  python -m src.preprocess
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLEAN_V3 = DATA / "clean_v3"
V3TR = DATA / "v3tr_orig"
_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        import yaml
        _CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf8"))["preprocess"]
    return _CFG


def read_color(path):
    import cv2

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def sha256(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pixel_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def product_mask(image, empty=1):
    """역 Otsu 이진화의 가장 큰 연결 성분 = 제품 영역."""
    import cv2

    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.full_like(image, empty, dtype=np.uint8)
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def clean_v3(raw):
    import cv2
    from scipy import ndimage

    cfg = _cfg()
    chroma = raw.max(2).astype(int) - raw.min(2).astype(int)
    ring = chroma > 40
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    interior = ndimage.binary_fill_holes(ring) & ~ring
    lab, n = ndimage.label(interior)
    for i in range(1, n + 1):
        region = lab == i
        if region.sum() < 30:
            continue
        inner = region & ~ndimage.binary_erosion(region, iterations=2)
        outer = (ndimage.binary_dilation(region | ring, iterations=4)
                 & ~ndimage.binary_dilation(region | ring, iterations=1))
        a, b = np.median(gray[inner]), np.median(gray[outer])
        g = float(np.clip(b / max(a, 1.0), 1.1, 1.45)) if a > 0 else cfg["overlay_gain"]
        g = cfg["gain_shrink"] * g + (1 - cfg["gain_shrink"]) * cfg["overlay_gain"]
        gray[region] = np.clip(gray[region] * g, 0, 255)
    g8 = gray.astype(np.uint8)
    mask = cv2.dilate(ring.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    if mask.any():
        g8 = cv2.inpaint(g8, mask, 3, cv2.INPAINT_TELEA)
        g = g8.astype(np.float32)
        resid = g - cv2.GaussianBlur(g, (0, 0), 1.5)
        m = mask > 0
        shifts = ((0, 26), (0, -26), (26, 0), (-26, 0), (26, 26), (-26, -26), (26, -26), (-26, 26))
        overlap = [int((np.roll(m, s, axis=(0, 1)) & m).sum()) for s in shifts]
        dy, dx = shifts[int(np.argmin(overlap))]      # 표시 자리와 가장 덜 겹치는 방향(동률이면 앞선 방향)
        src = np.roll(resid, (-dy, -dx), axis=(0, 1))
        g[m] = np.clip(g[m] + src[m], 0, 255)
        g8 = g.astype(np.uint8)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)


def draw_mark(gray, x, y, rng):
    """검사기 표시 모사: 내부 덮개(×darken) + 색 테두리. 반환: BGR(유채색 테두리 포함)."""
    import cv2

    f = _cfg()["fake_mark"]
    g = gray.astype(np.float32)
    h2, s2 = f["interior"] // 2, f["side"] // 2
    g[max(0, y - h2):y + h2, max(0, x - h2):x + h2] *= f["darken"]
    bgr = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    color = (0, 0, 255) if rng.random() < f["red_ratio"] else (255, 0, 0)
    cv2.rectangle(bgr, (x - s2, y - s2), (x + s2, y + s2), color, f["thick"])
    return bgr


def _gray(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def mark_centres(stem):
    import cv2

    raw = read_color(DATA / "raw" / f"{stem}.bmp").astype(int)
    m = ((raw.max(2) - raw.min(2)) > 40).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((3, 3), np.uint8)))
    return [tuple(cen[i]) for i in range(1, n) if st[i][cv2.CC_STAT_AREA] >= 12]


def free_spots(gray, avoid, n, rng, min_dist=30, edge=18):
    import cv2

    dist = cv2.distanceTransform(product_mask(gray, empty=0), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dist > edge)
    spots = []
    for _ in range(400):
        if len(spots) == n or not len(ys):
            break
        i = rng.integers(len(ys))
        x, y = int(xs[i]), int(ys[i])
        if all(np.hypot(x - a, y - b) > min_dist for a, b in list(avoid) + spots):
            spots.append((x, y))
    return spots


def truth(stem):
    """원본 TXT(YOLO 중심·크기, 정규화)를 정규화 xyxy 박스로 (protocol common.truth 와 같은 연산)."""
    rows = [list(map(float, l.split())) for l in (DATA / "labels_orig" / f"{stem}.txt").read_text().splitlines() if l.strip()]
    a = np.array(rows).reshape(-1, 5)[:, 1:]
    return np.c_[a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2].tolist()


def _avoid(stem, gray):
    h, w = gray.shape
    return [((b[0] + b[2]) / 2 * w, (b[1] + b[3]) / 2 * h) for b in truth(stem)] + mark_centres(stem)


def split_rows():
    with open(DATA / "split.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def dev_stems():
    a = json.loads((DATA / "validation_split.json").read_text(encoding="utf8"))["schemes"]["t1"]["assignment"]
    return {s for s, f in a.items() if f != -1}


def build():
    """원본 해시 확인 → v3(435장) → v3tr(개발 356장) → 픽셀 해시 대조 → 학습 라벨 사본(원본 TXT)."""
    import cv2

    cfg = _cfg()
    lo, hi = cfg["train_marks_per_image"]
    record = json.loads((DATA / "v3_pixel_sha256.json").read_text(encoding="utf8"))
    dev = dev_stems()
    n_v3 = n_tr = 0
    for r in split_rows():
        s = r["stem"]
        raw_path = DATA / "raw" / f"{s}.bmp"
        assert sha256(raw_path) == r["raw_sha256"], f"원본이 다릅니다: {raw_path.name}"
        v3 = clean_v3(read_color(raw_path))
        g = _gray(v3)
        assert pixel_sha256(g) == record["v3"][s], f"v3 전처리 불일치: {s}"
        write_image(CLEAN_V3 / "images" / f"{s}.png", v3)
        n_v3 += 1
        if s not in dev:                 # 평가 영상: 검출기 입력(v3)만. 라벨은 없고 읽지 않는다
            continue
        label = (DATA / "labels_orig" / f"{s}.txt").read_text(encoding="utf8")
        (CLEAN_V3 / "labels").mkdir(parents=True, exist_ok=True)
        (CLEAN_V3 / "labels" / f"{s}.txt").write_text(label, encoding="utf8")
        rng = np.random.default_rng(int(r["raw_sha256"][:8], 16))
        for x, y in free_spots(g, _avoid(s, g), int(rng.integers(lo, hi + 1)), rng):
            g = _gray(clean_v3(draw_mark(g, int(x), int(y), rng)))
        assert pixel_sha256(g) == record["v3tr"][s], f"v3tr 전처리 불일치: {s}"
        write_image(V3TR / "images" / f"{s}.png", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        (V3TR / "labels").mkdir(parents=True, exist_ok=True)
        (V3TR / "labels" / f"{s}.txt").write_text(label, encoding="utf8")
        n_tr += 1
    rep = {"raw_sha256_checked": n_v3, "v3_images_pixel_hash_equal": n_v3, "v3tr_images_pixel_hash_equal": n_tr,
           "record": "data/v3_pixel_sha256.json (protocol preprocessing record, 2026-09-26)"}
    out = ROOT / "outputs" / "validation" / "PREPROCESS_REPORT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"v3 {n_v3}장 · v3tr {n_tr}장 생성, 원본 해시·픽셀 해시 모두 기록과 일치", flush=True)
    return rep


if __name__ == "__main__":
    build()
