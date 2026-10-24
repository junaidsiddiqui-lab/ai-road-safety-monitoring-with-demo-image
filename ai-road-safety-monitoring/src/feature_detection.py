"""Canny edges, ORB keypoints/descriptors, and BFMatcher (Hamming) feature matching."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
from src import image_processing as ip


def detect_canny(gray, sigma=0.33):
    """Canny with thresholds derived from the median brightness (no manual tuning per image)."""
    blur = ip.gaussian_blur(gray)
    v = float(np.median(blur))
    lo, hi = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    return cv2.Canny(blur, lo, hi), (lo, hi)


def detect_orb(gray, nfeatures=None):
    orb = cv2.ORB_create(nfeatures=nfeatures or config.ORB_FEATURES)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_descriptors(des1, des2, method="ratio", ratio=0.75):
    """BFMatcher with Hamming distance. method='ratio' (Lowe ratio test) or 'crosscheck'."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    if method == "crosscheck":
        return sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2), key=lambda m: m.distance)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    return sorted([p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance], key=lambda m: m.distance)


def geometric_inliers(kp1, kp2, matches, thresh=5.0):
    """Count matches consistent with ONE homography (RANSAC) - a proxy for correct matches."""
    if len(matches) < 4:
        return 0
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    return int(mask.sum()) if mask is not None else 0


def analyse_pair(path1, path2, max_side=800):
    """Full feature pipeline for two related images; saves figures and returns real measured numbers."""
    config.MATCH_DIR.mkdir(parents=True, exist_ok=True)
    imgs = [ip.resize_max(ip.load_image(p), max_side) for p in (path1, path2)]
    grays = [ip.apply_clahe(ip.gaussian_blur(ip.to_gray(i))) for i in imgs]
    res = {"images": [path1.name, path2.name], "per_image": []}
    kps, dess = [], []
    for p, img, g in zip((path1, path2), imgs, grays):
        edges, th = detect_canny(g)
        kp, des = detect_orb(g)
        kps.append(kp); dess.append(des)
        cv2.imwrite(str(config.MATCH_DIR / f"{p.stem}_canny.png"), edges)
        cv2.imwrite(str(config.MATCH_DIR / f"{p.stem}_orb_keypoints.png"),
                    cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
        res["per_image"].append({"image": p.name, "canny_thresholds": list(th), "edge_pixel_share": float((edges > 0).mean()),
                                 "orb_keypoints": len(kp), "descriptor_shape": list(des.shape) if des is not None else None,
                                 "descriptor_bits": 256 if des is not None else None})
    for method in ("crosscheck", "ratio"):
        m = match_descriptors(dess[0], dess[1], method)
        inl = geometric_inliers(kps[0], kps[1], m)
        res[method] = {"matches": len(m), "mean_hamming_distance": float(np.mean([x.distance for x in m])) if m else None,
                       "ransac_inliers": inl, "inlier_share": (inl / len(m)) if m else None}
        vis = cv2.drawMatches(imgs[0], kps[0], imgs[1], kps[1], m[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        out = config.MATCH_DIR / f"matches_{method}.png"
        cv2.imwrite(str(out), vis); res[method]["figure"] = out.name
    # presentation figure: edges + keypoints
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, (p, img, g) in enumerate(zip((path1, path2), imgs, grays)):
        axes[0, j].imshow(detect_canny(g)[0], cmap="gray"); axes[0, j].set_title(f"Canny edges: {p.name}"); axes[0, j].axis("off")
        axes[1, j].imshow(ip.bgr_to_rgb(cv2.drawKeypoints(img, kps[j], None, color=(0, 255, 0)))); axes[1, j].set_title(f"ORB keypoints ({len(kps[j])})"); axes[1, j].axis("off")
    fig.tight_layout(); fig.savefig(config.FIG_DIR / "edges_and_keypoints.png", dpi=110); plt.close(fig)
    return res
