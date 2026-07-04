#!/usr/bin/env python3
"""诊断脚本：保存检测到的每个候选区域的裁剪图和预处理变体"""
import cv2, numpy as np, sys
from plate_recognition import PlateRecognizer

def diagnose(image_path):
    r = PlateRecognizer(gpu=False)
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image"); return

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # Denoise
    img_denoised = cv2.bilateralFilter(img, 7, 50, 50)

    # Detect on both
    for src_name, src_img in [("original", img), ("denoised", img_denoised)]:
        cands = r.detect(src_img)
        print(f"\n{src_name}: {len(cands)} candidates")
        for i, c in enumerate(cands):
            x, y, bw, bh, color, area, ar = c[:7]
            roi, (x1, y1, x2, y2) = r.crop(src_img, c)
            if roi.size == 0:
                continue

            # Save crop
            fname = f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_crop.jpg"
            cv2.imwrite(fname, roi)
            print(f"  #{i+1} [{color} AR={ar:.2f}] ({x1},{y1})-({x2},{y2}) → {fname}")

            # Save preprocessing variants
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_gray.jpg", gray)

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_otsu.jpg", binary)

            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 15, 4)
            cv2.imwrite(f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_adaptive.jpg", adaptive)

            inverted = cv2.bitwise_not(roi)
            cv2.imwrite(f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_inverted.jpg", inverted)

            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(roi, -1, sharpen_kernel)
            cv2.imwrite(f"diag_{Path(image_path).stem}_{src_name}_cand{i+1}_sharp.jpg", sharpened)

    print("\nDone! Check diag_*.jpg files to see what regions are being detected.")

if __name__ == '__main__':
    from pathlib import Path
    if len(sys.argv) < 2:
        print("Usage: python diagnose_detection.py <image>")
        sys.exit(1)
    diagnose(sys.argv[1])
