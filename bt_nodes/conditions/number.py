from bt_core.nodes import ConditionNode
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
from bt_utils.ocr_manager import OCRManager
from bt_nodes.conditions.common import PREPROCESS_MODE_MAP


class NumberConditionNode(ConditionNode):
    NODE_TYPE = "NumberConditionNode"

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        try:
            region = self._get_effective_region(context)
            screenshot = self._get_region_image(context, region)
            if screenshot is None:
                return False

            from bt_utils.direction import SearchDirection
            search_direction = self.config.get("search_direction", "左上")
            direction = SearchDirection.VALUE_MAP.get(search_direction, SearchDirection.TOP_LEFT)

            preprocess_display = self.config.get("preprocess_mode", "默认")
            preprocess_mode = PREPROCESS_MODE_MAP.get(preprocess_display, "normal")
            extract_mode = self.config.get("extract_mode", "无规则")
            extract_pattern = self.config.get("extract_pattern", "")
            min_confidence = self.config.get_float("min_confidence", 50) / 100.0

            if region is None:
                # 全屏直接识别时整图会被OCR引擎缩小，小字识别率很差，因此全屏改为按文本块逐块识别
                success, value, all_text, position = OCRManager().search_number_full_screen(
                    screenshot,
                    extract_mode=extract_mode,
                    extract_pattern=extract_pattern,
                    search_direction=direction
                )
            else:
                success, value, all_text, position = OCRManager().recognize_number_with_position(
                    screenshot,
                    preprocess_mode=preprocess_mode,
                    extract_mode=extract_mode,
                    extract_pattern=extract_pattern,
                    min_confidence=min_confidence,
                    search_direction=direction
                )
                position = self._adjust_position(position, region)

            if not success or value is None:
                self._log_condition_result(False, f"无法识别数字 (文本: {all_text})")
                return False

            if position:
                self._save_position(context, position)

            value_key = self.config.get("value_key", "last_number_value")
            context.blackboard.set("last_number_value", value)
            if value_key and value_key != "last_number_value":
                context.blackboard.set(value_key, value)

            comparison = self.config.get("compare_mode", ">=")
            target_value, error = self._resolve_target_value(context)
            if target_value is None:
                self._log_condition_result(False, error)
                return False
            result = self._compare_value(value, comparison, target_value)

            if result:
                self._log_condition_result(True, extra_info=f"值: {value}")
                return True
            else:
                self._log_condition_result(False,
                    f"数值比较失败: {value} {comparison} {target_value}")
                return False
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"NumberConditionNode '{self.name}'")
            self._log_condition_result(False, "检测异常，详情见终端日志")
            return False

    def _resolve_target_value(self, context) -> Tuple[Optional[float], str]:
        """比较值：固定值或黑板变量（变量不存在或不是数字时返回 None 和原因）"""
        if self.config.get("threshold_mode", "fixed") != "variable":
            return self.config.get_float("threshold", 0), ""

        var_name = self.config.get("threshold_key", "")
        if not var_name:
            return None, "未设置比较变量名"
        if not context.blackboard.exists(var_name):
            return None, f"黑板变量 '{var_name}' 不存在"
        value = context.blackboard.get(var_name)
        try:
            return float(value), ""
        except (TypeError, ValueError):
            return None, f"黑板变量 '{var_name}' 的值 {value!r} 不是数字"

    def _compare_value(self, value: float, comparison: str, target_value: float) -> bool:
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        op = ops.get(comparison, lambda a, b: False)
        return op(value, target_value)
