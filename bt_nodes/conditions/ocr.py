from bt_core.nodes import ConditionNode
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
from bt_utils.ocr_manager import OCRManager
from bt_nodes.conditions.common import (
    PREPROCESS_MODE_MAP,
    PREPROCESS_MODE_REVERSE_MAP,
)


class OCRConditionNode(ConditionNode):
    NODE_TYPE = "OCRConditionNode"

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        try:
            keywords = self._get_keywords(context)
            if keywords is None:
                return False

            from bt_utils.direction import SearchDirection
            search_direction = self.config.get("search_direction", "左上")
            direction = SearchDirection.VALUE_MAP.get(search_direction, SearchDirection.TOP_LEFT)
            preprocess_display = self.config.get("preprocess_mode", "默认")
            preprocess_mode = PREPROCESS_MODE_MAP.get(preprocess_display, "normal")

            region = self._get_effective_region(context)
            screenshot = self._get_region_image(context, region)
            if screenshot is None:
                return False

            if region is None and keywords.strip():
                # 全屏直接识别时整图会被OCR引擎缩小，小字识别率很差，
                # 因此全屏改为按文本块逐块粗找 + 候选区域放大确认
                found, position, all_text = OCRManager().search_full_screen(
                    screenshot, keywords,
                    preprocess_mode=preprocess_mode,
                    search_direction=direction
                )
            else:
                found, position, all_text = OCRManager().recognize(
                    screenshot, keywords,
                    preprocess_mode=preprocess_mode,
                    search_direction=direction
                )
                position = self._adjust_position(position, region)

            if found:
                self._save_position(context, position)
                self._log_condition_result(True)
                return True
            else:
                reason = f"未找到关键词: {keywords}"
                extra = f"识别到的文本: {all_text}" if all_text else None
                self._log_condition_result(False, reason, extra)
                return False
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"OCRConditionNode '{self.name}'")
            self._log_condition_result(False, "检测异常，详情见终端日志")
            return False

    def _get_keywords(self, context) -> Optional[str]:
        """取关键词：固定文本，或从黑板变量读取（由外部程序写入，如收货地区）

        Returns:
            关键词字符串；黑板变量未设置或为空时返回None（节点判定失败）
        """
        if self.config.get("keyword_source", "固定文本") != "黑板变量":
            return self.config.get("keywords", "")

        var_name = self.config.get("keyword_variable", "").strip()
        if not var_name:
            self._log_condition_result(False, "未设置关键词黑板变量名")
            return None
        value = context.blackboard.get(var_name)
        if value is None or str(value).strip() == "":
            self._log_condition_result(False, f"黑板变量 '{var_name}' 不存在或为空")
            return None
        return str(value).strip()
