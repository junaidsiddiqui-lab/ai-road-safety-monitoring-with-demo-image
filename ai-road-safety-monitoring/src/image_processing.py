"""Basic image processing: loading, inspection, grayscale, Gaussian blur, CLAHE, comparison plots."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


def load_image(path):
    """OpenCV loads images as BGR (not RGB). Returns the BGR array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_max(img, max_side=800):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def describe_image(img):
    h, w = img.shape[:2]
    return {"height": int(h), "width": int(w), "channels": int(img.shape[2]) if img.ndim == 3 else 1,
            "dtype": str(img.dtype), "min": int(img.min()), "max": int(img.max()),
            "pixel_0_0_BGR": img[0, 0].tolist(), "pixel_center_BGR": img[h // 2, w // 2].tolist()}


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, ksize=5, sigma=0):
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def apply_clahe(gray, clip_limit=2.0, tile=(8, 8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(gray)


def process_image(path, out_dir=None, max_side=800):
    """Run the full basic pipeline on one image, save the stages and a comparison figure."""
    out_dir = out_dir or config.PROC_IMG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    img = resize_max(load_image(path), max_side)
    gray = to_gray(img); blur = gaussian_blur(gray); clahe = apply_clahe(blur)
    for name, arr in [("original", img), ("gray", gray), ("blur", blur), ("clahe", clahe)]:
        cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), arr)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (t, a, cm) in zip(axes, [("Original (RGB)", bgr_to_rgb(img), None), ("Grayscale", gray, "gray"),
                                      ("Gaussian blur 5x5", blur, "gray"), ("CLAHE (clip 2.0, 8x8)", clahe, "gray")]):
        ax.imshow(a, cmap=cm); ax.set_title(t); ax.axis("off")
    fig.suptitle(f"Image processing steps: {path.name}"); fig.tight_layout()
    comp = config.FIG_DIR / f"image_processing_{stem}.png"
    fig.savefig(comp, dpi=110); plt.close(fig)
    return {"image": path.name, **describe_image(load_image(path)), "resized_to": list(gray.shape[::-1]),
            "gray_mean_before_clahe": float(blur.mean()), "gray_std_before_clahe": float(blur.std()),
            "gray_std_after_clahe": float(clahe.std()), "comparison_figure": comp.name}
