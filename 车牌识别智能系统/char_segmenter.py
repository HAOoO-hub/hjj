#char_segments.py
#功能：实现通过解析指定的ccpd_image_path=...
# 然后根据车牌位置分割出字符并保存到save_dir=r"E:\hjj_II\test result"
import cv2
import numpy as np
import os
from pathlib import Path


# ==================== 复用CCPD解析核心逻辑（仅保留车牌位置解析） ====================
class CCPDParser:
    """仅保留CCPD文件名解析和车牌裁剪的核心功能"""
    PROVINCE = [
        "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
        "新", "警", "学", "O"
    ]

    ALPHABET = [
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]

    @staticmethod
    def parse_filename(filename):
        """解析CCPD文件名，提取车牌位置和字符索引"""
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('-')

        if len(parts) < 6:
            return None

        # 解析车牌边界框（精确位置）
        bbox_part = parts[2]
        bbox_coords = bbox_part.replace('&', '_').split('_')
        try:
            x1 = int(bbox_coords[0])
            y1 = int(bbox_coords[1])
            x2 = int(bbox_coords[2])
            y2 = int(bbox_coords[3])
        except (IndexError, ValueError):
            return None
        bbox = (x1, y1, x2, y2)

        # 解析车牌角点（用于透视矫正）
        corners_part = parts[3]
        corners_coords = corners_part.replace('&', '_').split('_')
        corners = []
        try:
            for i in range(0, 8, 2):
                x = int(corners_coords[i])
                y = int(corners_coords[i + 1])
                corners.append((x, y))
        except (IndexError, ValueError):
            corners = []

        # 解析车牌字符索引
        plate_info_part = parts[4]
        plate_chars = plate_info_part.split('_')
        ground_truth = CCPDParser.indices_to_plate(plate_chars)

        return {
            'bbox': bbox,  # 车牌在原图中的精确位置 (x1,y1,x2,y2)
            'corners': corners,  # 车牌四个角点
            'ground_truth': ground_truth  # 真实车牌号
        }

    @staticmethod
    def indices_to_plate(indices):
        """从索引转换为车牌号"""
        if len(indices) != 7:
            return "索引长度错误"
        try:
            plate_number = []
            # 省份
            province_idx = int(indices[0])
            plate_number.append(
                CCPDParser.PROVINCE[province_idx] if 0 <= province_idx < len(CCPDParser.PROVINCE) else "?")
            # 第二位字母
            letter_idx = int(indices[1])
            plate_number.append(CCPDParser.ALPHABET[letter_idx] if 0 <= letter_idx < len(CCPDParser.ALPHABET) else "?")
            # 后五位
            for i in range(2, 7):
                char_idx = int(indices[i])
                plate_number.append(CCPDParser.ALPHABET[char_idx] if 0 <= char_idx < len(CCPDParser.ALPHABET) else "?")
            return ''.join(plate_number)
        except ValueError:
            return "索引解析错误"

    @staticmethod
    def extract_plate(image, file_info, padding=5):
        """根据解析的位置裁剪并矫正车牌（仅裁剪车牌区域，无多余背景）"""
        x1, y1, x2, y2 = file_info['bbox']
        # 少量padding（CCPD位置已精确，无需多补）
        h, w = image.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        # 裁剪车牌区域
        plate_region = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # 透视矫正（提升分割精度）
        if file_info['corners'] and len(file_info['corners']) == 4:
            try:
                adjusted_corners = [(cx - x1_pad, cy - y1_pad) for (cx, cy) in file_info['corners']]
                plate_corrected = CCPDParser.perspective_correction(plate_region, adjusted_corners)
                if plate_corrected is not None and plate_corrected.size > 0:
                    return plate_corrected
            except:
                pass
        return plate_region

    @staticmethod
    def perspective_correction(image, corners):
        """透视矫正为标准车牌尺寸（440x140）"""
        if len(corners) != 4:
            return None
        # 正确排序角点（左上、右上、右下、左下）- 修复原排序逻辑缺陷
        corners = np.array(corners, dtype=np.float32)
        # 计算中心点
        center = np.mean(corners, axis=0)
        # 按角度排序
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        sorted_corners = corners[sorted_idx].tolist()

        src_pts = np.array(sorted_corners, dtype=np.float32)
        # 标准车牌尺寸
        dst_pts = np.array([[0, 0], [439, 0], [439, 139], [0, 139]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (440, 140))


# ==================== HyperLPR风格字符分割核心（解决黑色问题） ====================
class HyperLPRCharSegmenter:
    """基于HyperLPR像素投影法的自适应字符分割器（解决黑色分割结果）"""

    def __init__(self):
        self.parser = CCPDParser()
        # HyperLPR核心参数
        self.plate_height = 140  # 标准车牌高度
        self.char_min_width = 15  # 字符最小宽度
        self.char_max_width = 80  # 字符最大宽度
        self.blank_threshold = 0.05  # 空白列阈值

    def read_image(self, image_path):
        """读取图片（支持中文路径）"""
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise Exception("图片读取失败")
            return img
        except Exception as e:
            print(f"❌ 读取图片错误: {e}")
            return None

    def preprocess_plate(self, plate_img):
        """HyperLPR风格预处理（解决黑色问题的核心）"""
        # 1. 统一缩放到标准高度，保持宽高比
        h, w = plate_img.shape[:2]
        scale = self.plate_height / h
        plate_img = cv2.resize(plate_img, (int(w * scale), self.plate_height))

        # 2. 灰度化 + 高斯模糊去噪
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. 自适应二值化（替代固定阈值，解决光照不均导致的黑色问题）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 2
        )

        # 4. 形态学操作：去除噪点，连接字符断裂
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2)

        return plate_img, binary

    def get_char_regions_by_projection(self, binary_plate):
        """
        HyperLPR核心：水平投影法分割字符（自适应间距）
        原理：统计每列的白色像素数，空白列作为字符分隔
        """
        h, w = binary_plate.shape
        char_regions = []

        # 1. 计算列投影（每列的白色像素数）
        col_projection = np.sum(binary_plate, axis=0) / 255  # 归一化到0~h

        # 2. 找出有效列（非空白列）
        valid_cols = []
        for col in range(w):
            if col_projection[col] > h * self.blank_threshold:
                valid_cols.append(col)

        if not valid_cols:
            print("❌ 未检测到有效字符列（可能是二值化失败）")
            return []

        # 3. 分割字符区域（基于空白列）
        start_col = valid_cols[0]
        prev_col = valid_cols[0]

        for col in valid_cols[1:]:
            # 如果当前列和上一列间隔超过1，认为是字符分隔
            if col - prev_col > 1:
                # 计算字符宽度
                char_width = prev_col - start_col + 1
                if char_width >= self.char_min_width and char_width <= self.char_max_width:
                    char_regions.append((start_col, 0, prev_col + 1, h))
                start_col = col
            prev_col = col

        # 添加最后一个字符
        char_width = prev_col - start_col + 1
        if char_width >= self.char_min_width and char_width <= self.char_max_width:
            char_regions.append((start_col, 0, prev_col + 1, h))

        # 4. 补充到7个字符（适配车牌7位）
        while len(char_regions) < 7:
            # 均分剩余空间补充
            if char_regions:
                last_x2 = char_regions[-1][2]
                new_width = min(self.char_max_width, (w - last_x2) // (7 - len(char_regions)))
                if new_width < self.char_min_width:
                    new_width = self.char_min_width
                char_regions.append((last_x2, 0, last_x2 + new_width, h))
            else:
                # 无有效字符时，均分整个宽度
                char_width = w // 7
                for i in range(7):
                    char_regions.append((i * char_width, 0, (i + 1) * char_width, h))
                break

        # 5. 裁剪到7个字符，防止过多
        char_regions = char_regions[:7]

        # 6. 最后一个字符适配到右边缘
        if char_regions:
            char_regions[-1] = (char_regions[-1][0], 0, w, h)

        return char_regions

    def segment_chars(self, image_path, save_dir=r"E:\hjj_II\test result"):
        """
        完整流程：CCPD解析车牌位置 + HyperLPR自适应分割字符
        """
        # 1. 读取图片
        img = self.read_image(image_path)
        if img is None:
            return None

        # 2. 解析CCPD文件名
        filename = os.path.basename(image_path)
        file_info = self.parser.parse_filename(filename)
        if not file_info:
            print(f"❌ 无法解析CCPD文件名: {filename}")
            return None
        plate_num = file_info['ground_truth']
        print(f"✅ 解析到车牌号: {plate_num}")

        # 3. 裁剪并矫正车牌（仅保留车牌区域）
        plate_img = self.parser.extract_plate(img, file_info)
        if plate_img is None:
            print(f"❌ 裁剪车牌失败")
            return None
        print(f"✅ 裁剪车牌尺寸: {plate_img.shape}")

        # 4. HyperLPR风格预处理（解决黑色问题）
        plate_img_scaled, binary_plate = self.preprocess_plate(plate_img)

        # 5. 像素投影法自适应分割字符
        char_regions = self.get_char_regions_by_projection(binary_plate)
        if not char_regions:
            print(f"❌ 字符分割失败")
            return None
        print(f"✅ 分割出 {len(char_regions)} 个字符区域")

        # 6. 创建保存目录
        save_path = os.path.join(save_dir, plate_num)
        os.makedirs(save_path, exist_ok=True)

        # 7. 保存分割结果（解决黑色问题）
        plate_with_box = plate_img_scaled.copy()
        for i, (x1, y1, x2, y2) in enumerate(char_regions):
            # 绘制字符框
            cv2.rectangle(plate_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 裁剪字符（使用缩放后的车牌图，避免尺寸问题）
            char_img = plate_img_scaled[y1:y2, x1:x2]

            # 字符图预处理（确保不是黑色）
            if char_img.mean() < 10:  # 几乎全黑时，使用二值化图
                char_img = binary_plate[y1:y2, x1:x2]
                char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)

            # 保存字符（命名：字符位置_字符内容.jpg）
            char_name = plate_num[i] if i < len(plate_num) else f"char_{i + 1}"
            char_save_path = os.path.join(save_path, f"{i + 1}_{char_name}.jpg")
            # 保存（支持中文路径）
            cv2.imencode('.jpg', char_img)[1].tofile(char_save_path)
            print(f"   保存字符{i + 1}: {char_save_path}")

        # 保存带框的车牌和二值化图
        cv2.imencode('.jpg', plate_with_box)[1].tofile(os.path.join(save_path, "plate_with_box.jpg"))
        cv2.imencode('.jpg', binary_plate)[1].tofile(os.path.join(save_path, "plate_binary.jpg"))

        print(f"✅ 所有字符已保存至: {save_path}")
        return {
            'plate_number': plate_num,
            'plate_image': plate_img_scaled,
            'char_regions': char_regions,
            'binary_plate': binary_plate,
            'save_directory': save_path
        }


# ==================== 测试调用示例 ====================
if __name__ == "__main__":
    # 初始化分割器
    segmenter = HyperLPRCharSegmenter()

    # 替换为你的CCPD图片路径
    ccpd_image_path = r"E:\hjj_II\testdata\II\0318342911878-89_83-153&576_454&681-471&671_160&690_156&586_467&567-0_0_8_32_30_26_5-125-46.jpg"

    # 执行分割
    result = segmenter.segment_chars(
        image_path=ccpd_image_path,
        save_dir= r"E:\hjj_II\test result"
    )

    if result:
        print(f"\n分割结果汇总：")
        print(f"车牌号: {result['plate_number']}")
        print(f"字符区域坐标: {result['char_regions']}")
        print(f"保存路径: {result['save_directory']}")