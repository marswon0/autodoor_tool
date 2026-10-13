from bt_core.nodes import ConditionNode
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
from bt_utils.image_processor import ImageProcessor


class ColorConditionNode(ConditionNode):
    NODE_TYPE = "ColorConditionNode"

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        try:
            region = self._get_effective_region(context)
            screenshot = self._get_region_image(context, region)
            if screenshot is None:
                return False

            target_color = self._parse_color(self.config.get("target_color", None))
            tolerance = self.config.get_int("tolerance", 30)
            match_mode = self.config.get("match_mode", "any")
            min_pixels = self.config.get_int("min_pixels", 1)
            match_ratio = self.config.get_float("color_match_threshold", 0.9)

            found, position = ImageProcessor.find_color(
                screenshot, target_color, tolerance,
                match_mode=match_mode, min_pixels=min_pixels, match_ratio=match_ratio
            )

            if found:
                actual_position = self._adjust_position(position, region)
                self._save_position(context, actual_position)
                self._log_condition_result(True)
                return True
            else:
                self._log_condition_result(False,
                    f"未找到匹配颜色 RGB{target_color} (容差: {tolerance})")
                return False
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"ColorConditionNode '{self.name}'")
            self._log_condition_result(False, "检测异常，详情见终端日志")
            return False
