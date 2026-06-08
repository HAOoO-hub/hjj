# image_resizer_for_hyperlpr.py
# 功能：将图片统一调整到CCPD标准尺寸720×1160
import os
import cv2
import numpy as np
import shutil
import datetime


def resize_images_to_ccpd_standard(input_dir, output_dir):
    """
    将图片统一调整到CCPD标准尺寸720×1160

    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出图片文件夹路径
    :return: 处理成功的图片数量
    """
    print("=" * 60)
    print("🖼️  图片规格调整程序")
    print("=" * 60)
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"目标尺寸：720×1160 像素 (CCPD数据集标准)")
    print("=" * 60)

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        return 0

    # 确保输出目录存在（清空已存在的内容）
    if os.path.exists(output_dir):
        print(f"⚠️  输出目录已存在，清空内容...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 获取所有图片文件
    image_files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file)

    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到图片文件")
        print(f"   支持的格式：{', '.join(image_extensions)}")
        return 0

    print(f"📁 找到 {len(image_files)} 张图片：")
    for i, img_file in enumerate(image_files, 1):
        print(f"   {i:2d}. {img_file}")

    print("\n" + "=" * 60)
    print("开始处理图片...")
    print("=" * 60)

    # 处理计数器
    success_count = 0
    process_log = []

    # CCPD标准尺寸
    TARGET_WIDTH = 720
    TARGET_HEIGHT = 1160

    # 处理每张图片
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_dir, img_file)

        print(f"\n📋 处理第 {idx}/{len(image_files)} 张：{img_file}")

        try:
            # 读取图片（支持中文路径）
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                print(f"   ❌ 读取失败：{img_file}")
                process_log.append(f"{idx:03d}. {img_file} ❌ 读取失败")
                continue

            original_height, original_width = img.shape[:2]
            print(f"   ✅ 原始尺寸：{original_width}×{original_height}")

            # 生成输出文件名（自动编号）
            output_filename = f"{idx}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # 调整图片到目标尺寸（保持宽高比，添加黑边）
            processed_img = resize_with_black_border(img, TARGET_WIDTH, TARGET_HEIGHT)

            # 保存处理后的图片
            cv2.imencode('.jpg', processed_img)[1].tofile(output_path)

            # 验证处理后的尺寸
            processed_height, processed_width = processed_img.shape[:2]

            print(f"   ✅ 处理后尺寸：{processed_width}×{processed_height}")
            print(f"   ✅ 保存为：{output_filename}")

            # 记录处理信息
            log_entry = f"{idx:03d}. {img_file} → {output_filename} | {original_width}×{original_height} → {processed_width}×{processed_height}"
            process_log.append(log_entry)

            success_count += 1

        except Exception as e:
            print(f"   ❌ 处理失败：{e}")
            process_log.append(f"{idx:03d}. {img_file} ❌ 处理失败：{e}")

    # 保存处理日志
    save_process_log(output_dir, input_dir, process_log, success_count, len(image_files))

    # 输出统计信息
    print_summary(success_count, len(image_files), output_dir)

    return success_count


def resize_with_black_border(img, target_width, target_height):
    """
    保持宽高比调整图片大小，通过添加黑边填充到目标尺寸

    :param img: 原始图片
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 处理后的图片
    """
    original_height, original_width = img.shape[:2]

    # 计算缩放比例（保持宽高比）
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale = min(width_ratio, height_ratio)

    # 计算缩放后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 调整图片大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 创建黑色背景
    result_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算放置位置（居中）
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将调整后的图片放在黑色背景中央
    result_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return result_img


def save_process_log(output_dir, input_dir, process_log, success_count, total_count):
    """保存处理日志"""
    log_file = os.path.join(output_dir, "resize_log.txt")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("图片规格调整日志\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"输入目录：{input_dir}\n")
        f.write(f"输出目录：{output_dir}\n")
        f.write(f"目标尺寸：720×1160 像素 (CCPD标准)\n")
        f.write(f"处理时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("处理详情：\n")
        f.write("-" * 60 + "\n")

        for log in process_log:
            f.write(log + "\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write(f"处理统计：成功 {success_count}/{total_count} 张图片\n")
        f.write(f"成功率：{success_count / total_count * 100:.1f}%\n")


def print_summary(success_count, total_count, output_dir):
    """输出处理摘要"""
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)
    print(f"📊 处理统计：")
    print(f"   总图片数：{total_count}")
    print(f"   成功处理：{success_count}")
    print(f"   失败数量：{total_count - success_count}")
    print(f"   成功率：{success_count / total_count * 100:.1f}%")
    print(f"\n📁 输出目录：{output_dir}")
    print(f"📝 处理日志：{output_dir}\\resize_log.txt")
    print("\n📄 输出文件：")
    print("   1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg, ...")
    print(f"\n📏 所有输出图片尺寸：720×1160 像素")
    print("✨ 已准备好用于HyperLPR识别")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"E:\final"  # 原始图片目录
    output_directory = r"E:\hjj_II\no_filename_test"  # 处理后的图片目录

    # 执行图片规格调整
    resize_images_to_ccpd_standard(input_directory, output_directory)