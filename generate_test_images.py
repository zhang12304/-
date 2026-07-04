#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试用的模拟车牌图片
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_test_plate_image(output_path, plate_text="京A12345", bg_type="car"):
    """
    创建一张模拟车牌图片
    :param output_path: 输出路径
    :param plate_text: 车牌号
    :param bg_type: 背景类型 (car / plain)
    """
    print(f"正在生成测试图片，车牌号: {plate_text}，背景: {bg_type}")

    if bg_type == "car":
        # 创建一个模拟的汽车前脸背景
        img = np.ones((600, 800, 3), dtype=np.uint8) * 200

        # 画车身
        cv2.rectangle(img, (50, 100), (750, 550), (80, 80, 100), -1)

        # 画车头/保险杠
        cv2.rectangle(img, (50, 350), (750, 550), (60, 60, 80), -1)

        # 画车灯
        cv2.ellipse(img, (150, 250), (60, 40), 0, 0, 360, (220, 220, 200), -1)
        cv2.ellipse(img, (650, 250), (60, 40), 0, 0, 360, (220, 220, 200), -1)

        # 画进气格栅
        cv2.rectangle(img, (250, 260), (550, 360), (30, 30, 40), -1)

        # 画车牌（蓝色底）
        plate_x, plate_y = 200, 380
        plate_w, plate_h = 400, 100
        cv2.rectangle(img, (plate_x, plate_y),
                      (plate_x + plate_w, plate_y + plate_h),
                      (180, 60, 10), -1)  # 蓝色底

        # 画白色边框
        cv2.rectangle(img, (plate_x, plate_y),
                      (plate_x + plate_w, plate_y + plate_h),
                      (255, 255, 255), 3)

    else:
        # 纯色背景
        img = np.ones((300, 600, 3), dtype=np.uint8) * 128
        # 蓝色车牌
        plate_x, plate_y = 100, 100
        plate_w, plate_h = 400, 100
        cv2.rectangle(img, (plate_x, plate_y),
                      (plate_x + plate_w, plate_y + plate_h),
                      (180, 60, 10), -1)
        cv2.rectangle(img, (plate_x, plate_y),
                      (plate_x + plate_w, plate_y + plate_h),
                      (255, 255, 255), 3)

    # 使用 PIL 写中文字符
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试加载中文字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 48)
                print(f"使用字体: {fp}")
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()
        print("使用默认字体")

    # 写字
    text_color = (255, 255, 255)  # 白色字
    text_x = plate_x + 20
    text_y = plate_y + 25
    draw.text((text_x, text_y), plate_text, font=font, fill=text_color)

    # 转换回 OpenCV 格式并保存
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_result)
    print(f"测试图片已保存至: {output_path}")
    return output_path


if __name__ == '__main__':
    # 生成几张测试图片
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # 蓝牌测试图
    create_test_plate_image(
        os.path.join(output_dir, "test_plate_blue.jpg"),
        plate_text="京A12345",
        bg_type="car"
    )

    # 纯车牌测试图
    create_test_plate_image(
        os.path.join(output_dir, "test_plate_simple.jpg"),
        plate_text="沪B67890",
        bg_type="plain"
    )

    # 新能源绿牌
    create_test_plate_image(
        os.path.join(output_dir, "test_plate_green.jpg"),
        plate_text="粤BD12345",
        bg_type="car"
    )

    print("\n✅ 所有测试图片生成完毕！")
    print("运行以下命令进行识别测试：")
    print(f"  python {os.path.join(output_dir, 'plate_recognition.py')} test_plate_blue.jpg")
