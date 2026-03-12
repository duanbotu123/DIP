from pathlib import Path

import cv2
import numpy as np

from run_global_transform import apply_transform
from run_point_transform import point_guided_deformation


def make_demo_image(height: int = 320, width: int = 480) -> np.ndarray:
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    for y in range(0, height, 20):
        cv2.line(img, (0, y), (width - 1, y), (220, 220, 220), 1)
    for x in range(0, width, 20):
        cv2.line(img, (x, 0), (x, height - 1), (220, 220, 220), 1)

    cv2.rectangle(img, (80, 70), (220, 220), (80, 130, 255), -1)
    cv2.circle(img, (330, 150), 65, (255, 120, 100), -1)
    cv2.ellipse(img, (250, 255), (110, 40), 0, 0, 360, (120, 200, 120), -1)
    cv2.putText(img, "ImageWarping", (120, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def save_global_results(img: np.ndarray, out_dir: Path) -> None:
    variants = [
        ("global_input", img, None),
        ("global_scale_1p4", None, dict(scale=1.4, rotation=0, translation_x=0, translation_y=0, flip_horizontal=False)),
        ("global_rotate_35", None, dict(scale=1.0, rotation=35, translation_x=0, translation_y=0, flip_horizontal=False)),
        ("global_translate", None, dict(scale=1.0, rotation=0, translation_x=70, translation_y=-40, flip_horizontal=False)),
        ("global_flip", None, dict(scale=1.0, rotation=0, translation_x=0, translation_y=0, flip_horizontal=True)),
        ("global_combo", None, dict(scale=1.2, rotation=-22, translation_x=40, translation_y=30, flip_horizontal=True)),
    ]

    for name, fixed_img, params in variants:
        if fixed_img is not None:
            out = fixed_img
        else:
            out = apply_transform(img, **params)
        cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def save_point_results(img: np.ndarray, out_dir: Path) -> None:
    h, w = img.shape[:2]
    source_pts = np.array(
        [
            [80, 70],
            [220, 70],
            [80, 220],
            [220, 220],
            [330, 85],
            [330, 215],
            [250, 255],
            [360, 255],
        ],
        dtype=np.float32,
    )

    target_pts = np.array(
        [
            [60, 95],
            [240, 55],
            [95, 245],
            [235, 205],
            [315, 95],
            [350, 215],
            [235, 245],
            [375, 272],
        ],
        dtype=np.float32,
    )

    warped = point_guided_deformation(img, source_pts, target_pts, alpha=1.0)

    vis = img.copy()
    for p in source_pts.astype(int):
        cv2.circle(vis, tuple(p), 3, (255, 0, 0), -1)
    for p in target_pts.astype(int):
        cv2.circle(vis, tuple(p), 3, (255, 0, 0), -1)
    for s, t in zip(source_pts.astype(int), target_pts.astype(int)):
        cv2.arrowedLine(vis, tuple(s), tuple(t), (0, 255, 0), 1)

    side_by_side = np.concatenate([img, warped], axis=1)
    cv2.imwrite(str(out_dir / "point_controls.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "point_warped.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "point_before_after.png"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = make_demo_image()
    save_global_results(img, out_dir)
    save_point_results(img, out_dir)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
