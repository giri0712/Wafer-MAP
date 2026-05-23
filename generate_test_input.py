"""
generate_test_input.py
Generates synthetic wafer map test images for each defect pattern type.
Run this before testing the demo UI if you don't have real wafer images.
"""

import numpy as np
import cv2
import os

OUTPUT_DIR = "test_images"
SIZE       = 256
CENTER     = SIZE // 2
RADIUS     = SIZE // 2 - 15
PASS_COLOR = (50, 200, 50)    # green (BGR)
FAIL_COLOR = (50,  50, 200)   # red   (BGR)


def base_wafer() -> np.ndarray:
    """Blank dark wafer background with circular boundary."""
    img = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 20
    return img


def draw_dies(img: np.ndarray, fail_mask_fn) -> np.ndarray:
    """Fill each die inside the wafer boundary using fail_mask_fn(x, y, dist)."""
    for y in range(SIZE):
        for x in range(SIZE):
            dx = x - CENTER
            dy = y - CENTER
            dist = np.sqrt(dx**2 + dy**2)
            if dist < RADIUS:
                color = FAIL_COLOR if fail_mask_fn(x, y, dist) else PASS_COLOR
                img[y, x] = color
    return img


def gen_edge_ring(img):
    return draw_dies(img, lambda x, y, d: d > RADIUS * 0.82)

def gen_center(img):
    return draw_dies(img, lambda x, y, d: d < RADIUS * 0.25)

def gen_donut(img):
    return draw_dies(img, lambda x, y, d: RADIUS * 0.45 < d < RADIUS * 0.70)

def gen_scratch(img):
    # Diagonal scratch
    return draw_dies(img, lambda x, y, d: abs((y - CENTER) - 0.8 * (x - CENTER)) < 8)

def gen_local_cluster(img):
    cx, cy = CENTER + 30, CENTER - 20
    return draw_dies(img, lambda x, y, d: np.sqrt((x-cx)**2 + (y-cy)**2) < 30)

def gen_near_full(img):
    return draw_dies(img, lambda x, y, d: d < RADIUS * 0.95)

def gen_random(img):
    rng = np.random.default_rng(42)
    mask = rng.random((SIZE, SIZE)) < 0.25
    return draw_dies(img, lambda x, y, d: mask[y, x])

def gen_edge_loc(img):
    return draw_dies(img, lambda x, y, d: d > RADIUS * 0.82 and x > CENTER)

def gen_clean(img):
    return draw_dies(img, lambda x, y, d: False)


GENERATORS = {
    "edge_ring":   gen_edge_ring,
    "center":      gen_center,
    "donut":       gen_donut,
    "scratch":     gen_scratch,
    "local_cluster": gen_local_cluster,
    "near_full":   gen_near_full,
    "random":      gen_random,
    "edge_loc":    gen_edge_loc,
    "clean":       gen_clean,
}


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Generating synthetic wafer map test images in '{OUTPUT_DIR}/' ...")

    for name, gen_fn in GENERATORS.items():
        img  = base_wafer()
        img  = gen_fn(img)
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"  ✓  {path}")

    print(f"\n[DONE] {len(GENERATORS)} test images generated in '{OUTPUT_DIR}/'")
    print("Upload any of these to the Streamlit app to test prediction.")
