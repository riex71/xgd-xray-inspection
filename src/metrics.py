"""위치 기준 채점 (원 protocol src/metrics.py 의 centre_matches 를 연산 그대로 옮김)."""
import numpy as np


def centre_matches(pred, truth, sizes, radius=3.0):
    """위치 기준 매칭: 예측 중심이 아직 짝이 없는 정답 중심에서 radius px 안이면 검출.

    약 10px 이물에서는 IoU 0.5 가 1~2px 박스 모양 차이에 좌우된다. 검사의 목적은 이물 위치를
    알리는 것이므로, 의료 영상 검출 평가처럼 중심 거리로 맞힘 여부를 판단한다.
    """
    result = []
    for stem in truth:
        w, h = sizes[stem]
        centres = [((g[0] + g[2]) / 2 * w, (g[1] + g[3]) / 2 * h)
                   for g in np.asarray(truth[stem]).reshape(-1, 4)]
        used = set()
        for p in sorted(pred.get(stem, []), key=lambda p: -p[4])[:30]:
            cx, cy = (p[0] + p[2]) / 2 * w, (p[1] + p[3]) / 2 * h
            near = [(np.hypot(cx - x, cy - y), k) for k, (x, y) in enumerate(centres)
                    if k not in used and np.hypot(cx - x, cy - y) <= radius]
            k = min(near)[1] if near else None
            if k is not None:
                used.add(k)
            result.append((float(p[4]), k is not None, stem, k))
    return sorted(result, key=lambda r: -r[0])
