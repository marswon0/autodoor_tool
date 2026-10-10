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
    REGION_MODE_DEFAULT = "dynamic"
    REGION_USE_LAST_POS_DEFAULT = False

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        if self._get_region_mode() == "dynamic":
            keywords = self._get_keywords(context)
            if keywords is None:
                return False
            if keywords.strip():
                return self._check_dynamic(context, keywords)

        try:
            screenshot = self._get_region_image(context)
            if screenshot is None:
                return False

            from bt_utils.direction import SearchDirection
            search_direction = self.config.get("search_direction", "左上")
            direction = SearchDirection.VALUE_MAP.get(search_direction, SearchDirection.TOP_LEFT)

            keywords = self._get_keywords(context)
            if keywords is None:
                return False
            preprocess_display = self.config.get("preprocess_mode", "默认")
            preprocess_mode = PREPROCESS_MODE_MAP.get(preprocess_display, "normal")

            found, position, all_text = OCRManager().recognize(
                screenshot, keywords,
                preprocess_mode=preprocess_mode,
                search_direction=direction
            )

            if found:
                if position:
                    region = self._get_effective_region(context)
                    if region:
                        position = (position[0] + region[0], position[1] + region[1])
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

    def _check_dynamic(self, context, keywords: str) -> bool:
        """动态区域检测：先在锚点区域识别；无锚点或锚点区域未找到时，全屏两阶段搜索

        全屏直接识别时整图会被OCR引擎缩小，小字识别率很差，
        因此全屏改为按文本块逐块粗找 + 候选区域放大确认（OCRManager.search_full_screen）。
        """
        try:
            from bt_utils.direction import SearchDirection
            search_direction = self.config.get("search_direction", "左上")
            direction = SearchDirection.VALUE_MAP.get(search_direction, SearchDirection.TOP_LEFT)
            preprocess_display = self.config.get("preprocess_mode", "默认")
            preprocess_mode = PREPROCESS_MODE_MAP.get(preprocess_display, "normal")
            ocr = OCRManager()

            region = self._resolve_dynamic_region(context)
            if region is not None:
                screenshot = context.get_screenshot(region)
                if screenshot is not None:
                    found, position, _ = ocr.recognize(
                        screenshot, keywords,
                        preprocess_mode=preprocess_mode,
                        search_direction=direction
                    )
                    if found:
                        if position:
                            position = (position[0] + region[0], position[1] + region[1])
                        self._save_position(context, position)
                        self._log_condition_result(True)
                        return True

            screenshot = context.get_screenshot(None)
            if screenshot is None:
                return False

            found, position, all_text = ocr.search_full_screen(
                screenshot, keywords,
                preprocess_mode=preprocess_mode,
                search_direction=direction
            )

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
