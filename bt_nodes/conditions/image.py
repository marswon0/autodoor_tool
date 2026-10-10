from bt_core.nodes import ConditionNode
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
from PIL import Image
import os
from bt_utils.image_processor import ImageProcessor


class ImageConditionNode(ConditionNode):
    NODE_TYPE = "ImageConditionNode"
    REGION_MODE_DEFAULT = "dynamic"
    REGION_USE_LAST_POS_DEFAULT = False

    # 类级别模板缓存，避免每次 tick 重新从磁盘加载
    # 值为 (文件修改时间, 文件大小, 图像)，文件被替换后自动重新加载
    _template_cache: Dict[str, Tuple[int, int, Image.Image]] = {}
    # 按显示器缩放调整后的模板缓存：{(路径, 缩放倍数): (原模板对象, 缩放后模板)}
    _scaled_template_cache: Dict[Tuple[str, float], Tuple[Image.Image, Image.Image]] = {}

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        try:
            resolved_path = self._resolve_template_path(context)
            if resolved_path is None:
                return False

            screenshot = self._get_region_image(context)
            if screenshot is None:
                return False

            template_path = self.config.get("template_path", "")
            if not os.path.exists(resolved_path):
                self._log_condition_result(False, f"模板文件不存在: {template_path}")
                return False

            template = self._load_template(resolved_path)
            if template is None:
                self._log_condition_result(False, f"无法加载模板文件: {template_path}")
                return False

            template, scale_note = self._scale_template_for_display(context, resolved_path, template)

            raw_threshold = self.config.get_float("threshold", 80)
            threshold = raw_threshold / 100.0 if raw_threshold > 1 else raw_threshold

            color_check = self.config.get_bool("color_check", False)
            if color_check:
                # 颜色阈值为百分比；兼容填写 0~1 小数
                color_threshold = self.config.get_float("color_threshold", 80)
                if color_threshold <= 1:
                    color_threshold *= 100
                found, position, confidence, color_score = ImageProcessor.find_template_with_color(
                    screenshot, template, threshold, color_threshold
                )
            else:
                found, position, confidence = ImageProcessor.find_template(
                    screenshot, template, threshold
                )

            if found:
                actual_position = self._adjust_position(position, context)
                self._save_position(context, actual_position)
                self._log_condition_result(True)
                return True
            elif color_check and color_score is not None:
                self._log_condition_result(
                    False,
                    f"形状匹配但颜色不符 (颜色阈值: {color_threshold:.0f}%, 最高颜色相似度: {color_score:.0f}%; "
                    f"形状阈值: {threshold}, 最高形状置信度: {confidence:.2f}){scale_note}"
                )
                return False
            else:
                self._log_condition_result(False, f"未找到匹配模板 (阈值: {threshold}, 最高置信度: {confidence:.2f}){scale_note}")
                return False
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"ImageConditionNode '{self.name}'")
            self._log_condition_result(False, "检测异常，详情见终端日志")
            return False

    def _load_template(self, resolved_path: str) -> Optional[Image.Image]:
        """加载模板图像（带缓存）

        Args:
            resolved_path: 模板文件的绝对路径

        Returns:
            PIL.Image 图像对象，加载失败返回None
        """
        try:
            stat = os.stat(resolved_path)
        except OSError:
            return None

        cached = self._template_cache.get(resolved_path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]

        try:
            # 完整读入内存后关闭文件，避免占用文件导致无法删除/覆盖
            with Image.open(resolved_path) as img:
                img.load()
                template = img.copy()
        except Exception:
            return None
        self._template_cache[resolved_path] = (stat.st_mtime_ns, stat.st_size, template)
        return template

    def _scale_template_for_display(self, context, resolved_path: str,
                                    template: Image.Image) -> Tuple[Image.Image, str]:
        """模板记录了截图时的显示器缩放比例时，按当前缩放比例调整模板尺寸

        未记录缩放比例的模板（旧模板、浏览导入的图片）原样返回。

        Returns:
            (用于匹配的模板, 日志说明；未缩放时为空字符串)
        """
        capture_scale = ImageProcessor.read_template_scale(template)
        if capture_scale is None:
            return template, ""

        current_scale = self._current_display_scale(context)
        factor = current_scale / capture_scale
        if abs(factor - 1.0) < 0.01:
            return template, ""

        key = (resolved_path, round(factor, 3))
        cached = self._scaled_template_cache.get(key)
        # 原模板对象变化（文件被替换后重新加载）时重新缩放
        if cached and cached[0] is template:
            scaled = cached[1]
        else:
            scaled = ImageProcessor.scale_template(template, factor)
            self._scaled_template_cache[key] = (template, scaled)

        note = (f"，模板缩放: {factor * 100:.0f}% "
                f"(截图时 {capture_scale * 100:.0f}%，当前 {current_scale * 100:.0f}%)")
        return scaled, note

    def _current_display_scale(self, context) -> float:
        """当前检测所在显示器的缩放比例：绑定窗口取窗口所在显示器；
        否则固定区域取区域中心所在显示器；其余情况取主显示器"""
        from bt_utils import dpi_awareness

        hwnd = context.get_bound_window() if hasattr(context, "get_bound_window") else None
        if hwnd:
            return dpi_awareness.get_window_dpi_scale(hwnd)

        if self._get_region_mode() == "fixed":
            region = self._parse_region(self.config.get("region", None))
            if region:
                return dpi_awareness.get_point_dpi_scale((region[0] + region[2]) // 2,
                                                         (region[1] + region[3]) // 2)

        return dpi_awareness.get_dpi_scale()

    @classmethod
    def clear_template_cache(cls):
        """清除模板缓存（项目切换时调用）"""
        cls._template_cache.clear()
        cls._scaled_template_cache.clear()

    def _resolve_template_path(self, context) -> Optional[str]:
        """解析模板路径

        Args:
            context: 执行上下文

        Returns:
            解析后的绝对路径，或None
        """
        template_path = self.config.get("template_path", "")
        if not template_path:
            self._log_condition_result(False, "未设置模板路径")
            return None

        if template_path.startswith("./") and hasattr(context, 'resolve_path'):
            return context.resolve_path(template_path)

        return template_path
