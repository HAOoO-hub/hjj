#hyper_location.py
#功能:通过把整车图片位置传入whole_car_path，调用hyperlpr进行车牌号的预测输出
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

# 新增：标注车牌位置并保存（无imshow依赖）
def mark_plate_and_save(img, plate_results, save_filename="hyperlpr_plate_marked.jpg"):
    """
    标注车牌位置并保存图片（替代imshow可视化）
    兼容两种坐标格式：1. 矩形框 [x1,y1,x2,y2]  2. 4个角点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param img: 原始图片
    :param plate_results: HyperLPR返回的识别结果
    :param save_filename: 保存的文件名（仅兼容旧调用，实际会被覆盖）
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

                # ========== 核心修改：替换文字绘制逻辑 ==========
                text_content = f"{plate_num} (置信度: {confidence:.4f})"
                img_copy = draw_chinese(img_copy, text_content, text_x, text_y)

                # 打印深度学习识别结果（清晰格式）
                print(f"\n📌 深度学习识别结果 {idx + 1}：")
                print(f"   车牌号：{plate_num}")
                print(f"   置信度：{confidence:.4f} (越高越准确)")
                print(f"   车牌坐标：{plate_pos}")
    else:
        # ========== 次要修改：未识别车牌时也用PIL绘中文 ==========
        img_copy = draw_chinese(img_copy, "未识别到任何车牌", 50, 50, 32)
        mark_info.append("未识别到任何车牌")
        print("\n❗ 未识别到任何车牌")

    # ========== 核心修改：仅改动这部分保存逻辑 ==========
    try:
        # 1. 定义目标目录并创建（不存在则自动创建）
        target_dir = r"E:\hjj_II\test result\hyper_location_result"
        os.makedirs(target_dir, exist_ok=True)

        # 2. 生成车牌号命名的文件名（过滤非法字符）
        if plate_results and len(plate_results) > 0:
            # 过滤Windows文件名非法字符
            plate_num_safe = plate_results[0][0].replace("\\", "").replace("/", "").replace(":", "").replace("*",
                                                                                                             "").replace(
                "?", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "")
            final_filename = f"{plate_num_safe}.jpg"
        else:
            final_filename = "未识别到车牌.jpg"

        # 3. 拼接最终保存路径
        save_path = os.path.join(target_dir, final_filename)

        # 4. 保存图片（支持中文路径）
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

# ==================== 测试1：识别整车图像 ====================
print("=" * 50)
print("测试1：识别整车图像")
print("=" * 50)
# 整车图像路径（使用原始字符串避免转义）
whole_car_path = r"E:\hjj_II\testdata\II\0318342911878-89_83-153&576_454&681-471&671_160&690_156&586_467&567-0_0_8_32_30_26_5-125-46.jpg"
whole_car_img = read_image(whole_car_path, "整车图像")

if whole_car_img is not None:
    try:
        # 调用HyperLPR识别
        results = hyperlpr.HyperLPR_plate_recognition(whole_car_img)
        print("\n📊 整车图像识别结果详情：")
        print(f"   结果类型: <class 'list'>")
        print(f"   结果长度: {len(results) if results else 0}")

        if results:
            print(f"   第一个结果: {results[0]}")
            print(f"   第一个结果长度: {len(results[0])}")
            for i, item in enumerate(results[0]):
                print(f"     元素{i}: {item}, 类型: {type(item)}")

            # 调用标注并保存函数（原传参保留，不影响）
            mark_plate_and_save(
                img=whole_car_img,
                plate_results=results,
                save_filename="hyperlpr_plate_marked.jpg"
            )
        else:
            # 即使未识别到也保存标注图片
            mark_plate_and_save(whole_car_img, results)

    except Exception as e:
        print(f"   ❌ 识别过程出错：{e}")
        import traceback

        traceback.print_exc()