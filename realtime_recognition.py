#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时摄像头车牌识别 v3
按 'q' 键退出，按 's' 键保存当前帧的识别截图，按空格键重新检测
"""

import cv2
import numpy as np
import time
import sys
import io
from plate_recognition import PlateRecognizer, PROVINCES, is_province, is_city, is_ok

# 修复 Windows 控制台 UTF-8 输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def realtime_recognition(camera_id=0):
    """实时摄像头车牌识别"""
    print("正在初始化实时车牌识别系统...")

    recognizer = PlateRecognizer(gpu=True)

    # 打开摄像头（Windows 优先 DirectShow）
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"错误：无法打开摄像头 (ID: {camera_id})")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("实时识别已启动！")
    print("   按 'q' 退出")
    print("   按 's' 保存当前识别结果")
    print("   按 '空格键' 重新检测当前帧")

    last_detect_time = 0
    detect_interval = 1.0  # 每秒检测一次
    current_plates = []
    detect_now = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        display_frame = frame.copy()
        current_time = time.time()

        # 定时/手动触发检测
        if detect_now or (current_time - last_detect_time > detect_interval):
            last_detect_time = current_time
            detect_now = False

            # 缩小图像加速检测
            h, w = frame.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                small_frame = cv2.resize(frame, (640, int(h * scale)))
            else:
                small_frame = frame

            # 使用新版 detect() 方法
            cands = recognizer.detect(small_frame)

            current_plates = []
            for c in cands[:5]:
                x, y, bw, bh, color, area, ar = c[:7]

                # 裁剪
                roi, (x1, y1, x2, y2) = recognizer.crop(small_frame, c)

                # 还原坐标到原始尺寸
                if scale != 1.0:
                    x1, y1 = int(x1 / scale), int(y1 / scale)
                    x2, y2 = int(x2 / scale), int(y2 / scale)
                    roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # OCR
                raw_texts = recognizer.ocr(roi)
                province_hits = recognizer.scan_province(roi)
                candidates = recognizer.extract(raw_texts, province_hits)
                candidates = [(p, s) for p, s in candidates if s > 0]

                if candidates:
                    best, score = candidates[0]
                    current_plates.append({
                        'number': best,
                        'score': score,
                        'position': (x1, y1, x2, y2),
                    })

        # 绘制结果
        for plate in current_plates:
            x1, y1, x2, y2 = plate['position']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = f"{plate['number']} ({plate['score']})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x1, y1 - lh - 10),
                          (x1 + lw, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 状态信息
        fps = 1 / (time.time() - current_time + 0.001)
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Plates: {len(current_plates)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Real-time License Plate Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"plate_capture_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"截图已保存: {filename}")
        elif key == 32:  # 空格键
            detect_now = True

    cap.release()
    cv2.destroyAllWindows()
    print("实时识别已停止。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='实时摄像头车牌识别')
    parser.add_argument('--camera', type=int, default=0, help='摄像头 ID')
    args = parser.parse_args()

    try:
        realtime_recognition(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\n用户中断")
