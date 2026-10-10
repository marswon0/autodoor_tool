import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict


class ImageProcessor:
    """图像处理器

    提供模板匹配、颜色检测和图像哈希识别功能。
    """

    # 模板哈希缓存：{(模板phash, hash_type): 模板哈希值}
    _template_hash_cache: Dict[str, str] = {}

    @staticmethod
    def find_template(source: Image.Image, template: Image.Image,
                      threshold: float = 0.8) -> Tuple[bool, Optional[Tuple[int, int]], float]:
        """模板匹配

        Args:
            source: 源图像
            template: 模板图像
            threshold: 匹配阈值

        Returns:
            (是否找到, 中心位置, 最高置信度) 元组。源图小于模板时返回 (False, None, 0.0)
        """
        source_array = np.array(source)
        template_array = np.array(template)

        # 统一转 RGB 三通道
        if len(source_array.shape) == 2:
            source_array = cv2.cvtColor(source_array, cv2.COLOR_GRAY2RGB)
        elif source_array.shape[2] == 4:
            source_array = cv2.cvtColor(source_array, cv2.COLOR_RGBA2RGB)

        if len(template_array.shape) == 2:
            template_array = cv2.cvtColor(template_array, cv2.COLOR_GRAY2RGB)
        elif template_array.shape[2] == 4:
            template_array = cv2.cvtColor(template_array, cv2.COLOR_RGBA2RGB)

        # 校验尺寸：源图必须 >= 模板
        sh, sw = source_array.shape[:2]
        th, tw = template_array.shape[:2]
        if sh < th or sw < tw:
            return False, None, 0.0

        source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)

        result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = template_gray.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return True, (center_x, center_y), max_val

        return False, None, max_val

    # 模板PNG中记录截图时显示器缩放比例的文本字段（可选，旧模板没有）
    TEMPLATE_SCALE_KEY = "autodoor_dpi_scale"

    @staticmethod
    def save_template(image: Image.Image, path: str, capture_scale: Optional[float] = None) -> None:
        """保存模板图像，可选在PNG内记录截图时的显示器缩放比例"""
        if capture_scale and path.lower().endswith(".png"):
            from PIL.PngImagePlugin import PngInfo
            info = PngInfo()
            info.add_text(ImageProcessor.TEMPLATE_SCALE_KEY, f"{capture_scale:g}")
            image.save(path, pnginfo=info)
        else:
            image.save(path)

    @staticmethod
    def read_template_scale(image: Image.Image) -> Optional[float]:
        """读取模板记录的截图缩放比例，没有记录或无效时返回None"""
        try:
            value = float(image.info.get(ImageProcessor.TEMPLATE_SCALE_KEY, ""))
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def scale_template(template: Image.Image, factor: float) -> Image.Image:
        """按比例缩放模板（factor≈1 时原样返回）"""
        if abs(factor - 1.0) < 0.01:
            return template
        size = (max(1, round(template.width * factor)), max(1, round(template.height * factor)))
        return template.resize(size, Image.LANCZOS)

    # 颜色校验：最多校验的形状候选数（形状相似、颜色不同的多个目标，如勾选/未勾选的复选框）
    COLOR_CHECK_MAX_CANDIDATES = 20
    # Lab 色度（√(a²+b²)）超过该值视为"有颜色"的像素
    COLOR_CHECK_CHROMA_MIN = 15.0

    @staticmethod
    def find_template_with_color(source: Image.Image, template: Image.Image,
                                 threshold: float = 0.8,
                                 color_threshold: float = 85.0
                                 ) -> Tuple[bool, Optional[Tuple[int, int]], float, Optional[float]]:
        """模板匹配 + 颜色校验（形状与颜色两个阈值需同时满足）

        按形状置信度从高到低逐个检查形状达标（>= threshold）的位置，
        返回第一个颜色相似度 >= color_threshold 的位置（即颜色达标中形状最好的）。

        颜色相似度 = 100 - 平均色差ΔE（CIELAB，下限0）。平均只统计模板或截图中"有颜色"的像素，
        避免小面积颜色变化被大面积白色/灰色背景稀释；两者都几乎无颜色时统计全部像素。

        Args:
            source: 源图像
            template: 模板图像
            threshold: 形状匹配阈值 (0~1)
            color_threshold: 颜色相似度阈值 (0~100)

        Returns:
            (是否找到, 中心位置, 最高形状置信度, 形状达标候选中的最高颜色相似度) 元组；
            没有形状达标的候选时颜色相似度为 None
        """
        source_rgb = np.array(source.convert("RGB"))
        template_rgb = np.array(template.convert("RGB"))

        sh, sw = source_rgb.shape[:2]
        th, tw = template_rgb.shape[:2]
        if sh < th or sw < tw:
            return False, None, 0.0, None

        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
        result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, best_shape, _, _ = cv2.minMaxLoc(result)

        template_lab = ImageProcessor._to_lab(template_rgb)
        best_color = None

        # minMaxLoc 每次取剩余最高分，候选按形状置信度从高到低
        for _ in range(ImageProcessor.COLOR_CHECK_MAX_CANDIDATES):
            _, shape_val, _, (x, y) = cv2.minMaxLoc(result)
            if shape_val < threshold:
                break

            patch_lab = ImageProcessor._to_lab(source_rgb[y:y + th, x:x + tw])
            color_score = ImageProcessor._color_similarity(template_lab, patch_lab)
            if color_score >= color_threshold:
                return True, (x + tw // 2, y + th // 2), best_shape, color_score
            if best_color is None or color_score > best_color:
                best_color = color_score

            # 抑制该候选附近区域，避免同一目标被重复选中
            x1, y1 = max(0, x - tw // 2), max(0, y - th // 2)
            result[y1:y + th // 2 + 1, x1:x + tw // 2 + 1] = -1.0

        return False, None, best_shape, best_color

    @staticmethod
    def _to_lab(rgb: np.ndarray) -> np.ndarray:
        """RGB(uint8) → CIELAB(float32，L 0~100)"""
        return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)

    @staticmethod
    def _color_similarity(template_lab: np.ndarray, patch_lab: np.ndarray) -> float:
        """颜色相似度 (0~100) = 100 - 平均ΔE76，只统计模板或截图中有颜色的像素"""
        delta_e = np.sqrt(np.sum((template_lab - patch_lab) ** 2, axis=2))

        chroma_t = np.hypot(template_lab[:, :, 1], template_lab[:, :, 2])
        chroma_p = np.hypot(patch_lab[:, :, 1], patch_lab[:, :, 2])
        mask = (chroma_t > ImageProcessor.COLOR_CHECK_CHROMA_MIN) | (chroma_p > ImageProcessor.COLOR_CHECK_CHROMA_MIN)
        # 有颜色的像素太少（灰白图标）时，统计全部像素
        if np.count_nonzero(mask) < max(20, int(mask.size * 0.02)):
            mean_delta = float(np.mean(delta_e))
        else:
            mean_delta = float(np.mean(delta_e[mask]))

        return max(0.0, 100.0 - mean_delta)

    @staticmethod
    def find_color(source: Image.Image, target_color: Tuple[int, int, int],
                   tolerance: int = 10, match_mode: str = "any",
                   min_pixels: int = 1, match_ratio: float = 0.9
                   ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """颜色检测

        Args:
            source: 源图像
            target_color: 目标颜色 (R, G, B)
            tolerance: 容差
            match_mode: 匹配模式，"any"=任一像素匹配即可，"all"=匹配像素占比需达到 match_ratio
            min_pixels: any 模式下最小匹配像素数
            match_ratio: all 模式下匹配像素占比阈值

        Returns:
            (是否找到, 中心位置) 元组。位置为匹配像素的中位数位置。
        """
        source_array = np.array(source)

        lower = np.array([max(0, c - tolerance) for c in target_color])
        upper = np.array([min(255, c + tolerance) for c in target_color])

        mask = cv2.inRange(source_array[:, :, :3], lower, upper)

        positions = np.where(mask > 0)
        match_count = len(positions[0])

        if match_mode == "all":
            total_pixels = mask.size
            ratio = match_count / total_pixels if total_pixels > 0 else 0
            if ratio < match_ratio:
                return False, None
        else:
            if match_count < min_pixels:
                return False, None

        if match_count > 0:
            # 使用中位数位置（与 ColorConditionNode 原逻辑一致）
            center_idx = match_count // 2
            center_x = int(positions[1][center_idx])
            center_y = int(positions[0][center_idx])
            return True, (center_x, center_y)

        return False, None

    @staticmethod
    def compute_phash(image: Image.Image, hash_size: int = 8) -> str:
        img = image.convert('L').resize((32, 32))
        img_array = np.array(img, dtype=np.float64)
        
        dct = cv2.dct(img_array)
        
        dct_low = dct[:hash_size, :hash_size]
        
        dct_low_flat = dct_low.flatten()
        dct_low_flat_no_dc = dct_low_flat[1:]
        median = np.median(dct_low_flat_no_dc)
        
        diff = dct_low > median
        
        return ''.join(['1' if b else '0' for b in diff.flatten()])

    @staticmethod
    def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
        """计算差异哈希

        Args:
            image: PIL.Image 图像
            hash_size: 哈希大小

        Returns:
            哈希字符串
        """
        img_array = np.array(image.convert('L').resize((hash_size + 1, hash_size)))
        
        diff = img_array[:, 1:] > img_array[:, :-1]
        
        return ''.join(['1' if b else '0' for b in diff.flatten()])

    @staticmethod
    def compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
        """计算平均哈希

        Args:
            image: PIL.Image 图像
            hash_size: 哈希大小

        Returns:
            哈希字符串
        """
        img_array = np.array(image.convert('L').resize((hash_size, hash_size)))
        
        avg = img_array.mean()
        
        diff = img_array > avg
        
        return ''.join(['1' if b else '0' for b in diff.flatten()])

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算汉明距离

        Args:
            hash1: 哈希字符串1
            hash2: 哈希字符串2

        Returns:
            汉明距离
        """
        if len(hash1) != len(hash2):
            return -1
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @staticmethod
    def _get_template_hash(template: Image.Image, hash_func, hash_type: str) -> str:
        """获取模板哈希（带缓存）

        使用模板自身的 phash 作为缓存键，避免同一模板重复计算哈希。
        """
        cache_key = f"{ImageProcessor.compute_phash(template)}_{hash_type}"
        if cache_key not in ImageProcessor._template_hash_cache:
            ImageProcessor._template_hash_cache[cache_key] = hash_func(template)
        return ImageProcessor._template_hash_cache[cache_key]

    @staticmethod
    def find_by_hash(source: Image.Image, templates: List[Image.Image],
                    threshold: float = 5, hash_type: str = "phash") -> Tuple[bool, Optional[Tuple[int, int]], Optional[int]]:
        """基于哈希的图像查找

        使用粗-细两阶段搜索：先以模板尺寸 1/4 为步长粗搜索，再在命中点周围逐像素精搜索。

        Args:
            source: 源图像
            templates: 模板图像列表
            threshold: 最大汉明距离阈值
            hash_type: 哈希类型 (phash/dhash/ahash)

        Returns:
            (是否找到, 中心位置, 最佳匹配索引) 元组。未找到时第三个元素为 None。
        """
        if not templates:
            return False, None, None

        if hash_type == "phash":
            hash_func = ImageProcessor.compute_phash
        elif hash_type == "dhash":
            hash_func = ImageProcessor.compute_dhash
        else:
            hash_func = ImageProcessor.compute_ahash

        source_array = np.array(source)
        h, w = source_array.shape[:2]

        best_match = None
        best_distance = threshold + 1
        best_index = None

        for idx, template in enumerate(templates):
            template_hash = ImageProcessor._get_template_hash(template, hash_func, hash_type)

            template_array = np.array(template)
            th, tw = template_array.shape[:2]

            if th > h or tw > w:
                continue

            # 阶段1：粗搜索（步长 = 模板尺寸的 1/4）
            coarse_step_y = max(1, th // 4)
            coarse_step_x = max(1, tw // 4)
            coarse_candidates = []

            for y in range(0, h - th + 1, coarse_step_y):
                for x in range(0, w - tw + 1, coarse_step_x):
                    region = source.crop((x, y, x + tw, y + th))
                    source_hash = hash_func(region)
                    distance = ImageProcessor.hamming_distance(source_hash, template_hash)

                    if distance <= threshold:
                        coarse_candidates.append((distance, x, y))

            # 阶段2：在粗搜索命中点周围逐像素精搜索
            if coarse_candidates:
                coarse_candidates.sort()  # 按距离排序，优先搜索最佳候选
                for _, cx, cy in coarse_candidates[:3]:
                    y_start = max(0, cy - coarse_step_y)
                    y_end = min(h - th + 1, cy + coarse_step_y + 1)
                    x_start = max(0, cx - coarse_step_x)
                    x_end = min(w - tw + 1, cx + coarse_step_x + 1)

                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            region = source.crop((x, y, x + tw, y + th))
                            source_hash = hash_func(region)
                            distance = ImageProcessor.hamming_distance(source_hash, template_hash)

                            if distance < best_distance:
                                best_distance = distance
                                best_match = (x + tw // 2, y + th // 2)
                                best_index = idx

                                if distance == 0:
                                    return True, best_match, best_index

        if best_match and best_distance <= threshold:
            return True, best_match, best_index

        return False, None, None
