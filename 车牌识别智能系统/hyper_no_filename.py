# hyper_no_filename.py
# 功能: 批量识别指定文件夹下的图片，调用hyperlpr进行车牌号的预测输出
import os
import hyperlpr
import cv2
import numpy as np
# ====== 新增1：导入PIL相关库（最小新增依赖） ======
from PIL import Image, ImageDraw, ImageFont


# 定义辅助函数：读取图片并校验
def read_image(image_path, desc):
    """
    读取图片并校验是否成功
    :param image_path: 图片路径
    :param desc: 图片描述（用于日志输出）
    :return: 读取成功返回图片数组，失败返回None
    """
    # 处理路径空格/中文问题（Windows下兼容）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 【{desc}】读取失败！请检查路径：{image_path}")
        print("   可能原因：文件不存在/路径错误/文件损坏/格式不支持（仅支持jpg/png/bmp）")
        return None
    print(f"✅ 【{desc}】读取成功，图片尺寸：{img.shape}")
    return img


# ====== 新增2：添加绘制中文的辅助函数（仅新增这一个函数） ======
def draw_chinese(img, text, x, y, font_size=24):
    """仅绘制中文的极简函数"""
    # 转PIL格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 加载Windows黑体（保证中文显示）
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    # 绘制白色背景+红色文字
    draw.rectangle([x, y - 28, x + len(text) * 20, y], fill=(255, 255, 255))
    draw.text((x, y - 24), text, font=font, fill=(255, 0, 0))
    # 转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# 修改：标注车牌位置并保存（支持批量处理）
def mark_plate_and_save(img, plate_results, original_filename, save_dir):
    """
    标注车牌位置并保存图片
    兼容两种坐标格式：1. 矩形框 [x1,y1,x2,y2]  2. 4个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param img: 原始图片
    :param plate_results: HyperLPR返回的识别结果
    :param original_filename: 原始文件名（不含路径）
    :param save_dir: 保存目录
    :return: 保存成功返回文件路径，失败返回None
    """
    # 复制图片避免修改原图
    img_copy = img.copy()
    # 定义绘制参数
    box_color = (0, 255, 0)  # 车牌框颜色：绿色
    line_thickness = 2  # 框线粗细

    # 记录标注信息
    mark_info = []

    if plate_results:
        for idx, result in enumerate(plate_results):
            if len(result) >= 3:
                plate_num = result[0]  # 车牌号
                confidence = result[1]  # 置信度
                plate_pos = result[2]  # 车牌坐标

                # ========== 兼容两种坐标格式 ==========
                plate_pos_np = np.array(plate_pos, dtype=np.int32)
                # 情况1：矩形框 [x1,y1,x2,y2] → 绘制矩形框
                if plate_pos_np.ndim == 1 and len(plate_pos_np) == 4:
                    x1, y1, x2, y2 = plate_pos_np
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), box_color, line_thickness)
                    text_x, text_y = x1, y1 - 10  # 文字位置
                    mark_info.append(
                        f"车牌{idx + 1}：{plate_num} | 位置：({x1},{y1})-({x2},{y2}) | 置信度：{confidence:.4f}")
                # 情况2：4个角点 → 绘制多边形
                elif plate_pos_np.ndim == 2 and len(plate_pos_np) == 4:
                    plate_corners = plate_pos_np
                    cv2.polylines(img_copy, [plate_corners], isClosed=True, color=box_color, thickness=line_thickness)
                    text_x, text_y = plate_corners[0][0], plate_corners[0][1] - 10  # 文字位置
                    mark_info.append(
                        f"车牌{idx + 1}：{plate_num} | 角点：{plate_corners.tolist()} | 置信度：{confidence:.4f}")
                else:
                    mark_info.append(f"车牌{idx + 1}：不支持的坐标格式 {plate_pos}")
                    continue

                # ========== 绘制中文文字 ==========
                text_content = f"{plate_num} (置信度: {confidence:.4f})"
                img_copy = draw_chinese(img_copy, text_content, text_x, text_y)

                # 打印深度学习识别结果（清晰格式）
                print(f"\n📌 深度学习识别结果 {idx + 1}：")
                print(f"   车牌号：{plate_num}")
                print(f"   置信度：{confidence:.4f} (越高越准确)")
                print(f"   车牌坐标：{plate_pos}")
    else:
        # 未识别车牌时用PIL绘制中文
        img_copy = draw_chinese(img_copy, "未识别到任何车牌", 50, 50, 32)
        mark_info.append("未识别到任何车牌")
        print("\n❗ 未识别到任何车牌")

    # ========== 保存逻辑 ==========
    try:
        # 1. 确保目标目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 2. 生成新的文件名（基于识别结果或原始文件名）
        if plate_results and len(plate_results) > 0:
            # 使用第一个识别到的车牌号作为文件名
            plate_num = plate_results[0][0]
            # 过滤Windows文件名非法字符
            plate_num_safe = plate_num.replace("\\", "").replace("/", "").replace(":", "").replace("*", "").replace(
                "?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")
            # 保留原始文件扩展名
            original_name_without_ext = os.path.splitext(original_filename)[0]
            final_filename = f"{original_name_without_ext}_{plate_num_safe}.jpg"
        else:
            # 未识别到车牌时，使用原始文件名
            original_name_without_ext = os.path.splitext(original_filename)[0]
            final_filename = f"{original_name_without_ext}_未识别到车牌.jpg"

        # 3. 拼接最终保存路径
        save_path = os.path.join(save_dir, final_filename)

        # 4. 如果文件名已存在，添加序号避免覆盖
        counter = 1
        while os.path.exists(save_path):
            name_without_ext = os.path.splitext(final_filename)[0]
            ext = os.path.splitext(final_filename)[1]
            save_path = os.path.join(save_dir, f"{name_without_ext}_{counter}{ext}")
            counter += 1

        # 5. 保存图片（支持中文路径）
        cv2.imencode('.jpg', img_copy)[1].tofile(save_path)
        print(f"\n✅ 标注后的图片已保存至：{save_path}")

        # 打印标注信息汇总
        print("📋 车牌标注信息汇总：")
        for info in mark_info:
            print(f"   - {info}")

        return save_path
    except Exception as e:
        print(f"❌ 保存图片失败：{e}")
        return None


# ==================== 兼容补丁：修复estimateRigidTransform缺失问题 ====================
if not hasattr(cv2, 'estimateRigidTransform'):
    def estimateRigidTransform(src, dst, fullAffine):
        M, inliers = cv2.estimateAffinePartial2D(src, dst)
        return M


    cv2.estimateRigidTransform = estimateRigidTransform


# ==================== 主程序：批量处理文件夹下的图片 ====================
def batch_process_images(input_dir, output_dir):
    """
    批量处理文件夹下的所有图片
    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出结果文件夹路径
    """
    print("=" * 50)
    print("批量识别图片程序")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print("=" * 50)

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录下的所有图片文件（支持jpg, png, bmp格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
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
        return

    print(f"📁 找到 {len(image_files)} 张图片：")
    for i, img_file in enumerate(image_files, 1):
        print(f"   {i}. {img_file}")

    print("\n" + "=" * 50)
    print("开始批量处理...")
    print("=" * 50)

    # 处理计数器
    processed_count = 0
    success_count = 0

    # 遍历所有图片文件
    for img_file in image_files:
        processed_count += 1
        img_path = os.path.join(input_dir, img_file)

        print(f"\n{'=' * 30}")
        print(f"处理第 {processed_count}/{len(image_files)} 张图片：{img_file}")
        print(f"{'=' * 30}")

        # 读取图片
        img = read_image(img_path, f"图片 {img_file}")

        if img is not None:
            try:
                # 调用HyperLPR识别
                print(f"\n🔍 正在识别 {img_file} ...")
                results = hyperlpr.HyperLPR_plate_recognition(img)

                print(f"\n📊 {img_file} 识别结果详情：")
                print(f"   结果类型: <class 'list'>")
                print(f"   结果长度: {len(results) if results else 0}")

                if results:
                    print(f"   第一个结果: {results[0]}")
                    print(f"   第一个结果长度: {len(results[0])}")
                    for i, item in enumerate(results[0]):
                        print(f"     元素{i}: {item}, 类型: {type(item)}")

                # 调用标注并保存函数
                save_path = mark_plate_and_save(
                    img=img,
                    plate_results=results,
                    original_filename=img_file,
                    save_dir=output_dir
                )

                if save_path:
                    success_count += 1

            except Exception as e:
                print(f"   ❌ 识别过程出错：{e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ❌ 跳过 {img_file}，因为图片读取失败")

    # 输出处理统计
    print("\n" + "=" * 50)
    print("批量处理完成！")
    print("=" * 50)
    print(f"📊 处理统计：")
    print(f"   总图片数：{len(image_files)}")
    print(f"   成功处理：{success_count}")
    print(f"   失败数量：{len(image_files) - success_count}")
    print(f"\n📁 所有标注结果已保存到：{output_dir}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"E:\hjj_II\testdata\no_filename_test" # 包含5张图片的文件夹
    output_directory = r"E:\hjj_II\test result\no_filename_result"  # 输出结果文件夹

    # 执行批量处理
    batch_process_images(input_directory, output_directory)

    #让程序返回"运行成功"
    "运行成功"

    # 原来的单张图片测试代码（可选保留，已注释掉）
    '''
    print("=" * 50)
    print("测试1：识别整车图像")
    print("=" * 50)
    # 整车图像路径（使用原始字符串避免转义）
    whole_car_path = r"E:\hjj_II\testdata\II\0318342911878-89_83-153&576_454&681-471&671_160&690_156&586_467&567-0_0_8_32_30_26_5-125-46.jpg"
    whole_car_img = read_image(whole_car_path, "整车图像")
    # ... 原来的单张测试代码 ...
    '''