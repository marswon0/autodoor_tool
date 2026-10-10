from bt_core.nodes import ConditionNode
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
from bt_utils.log_manager import LogManager
from bt_utils.ocr_manager import OCRManager
from bt_nodes.conditions.common import PREPROCESS_MODE_MAP


EXTRACT_MODE_MAP = {
    "全部": "all",
    "关键词": "keywords",
    "all": "all",
    "keywords": "keywords",
}


class TextExtractNode(ConditionNode):
    NODE_TYPE = "TextExtractNode"
    REGION_MODE_DEFAULT = "dynamic"
    REGION_USE_LAST_POS_DEFAULT = False

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)

    def _check_condition(self, context) -> bool:
        try:
            screenshot = self._get_region_image(context)
            if screenshot is None:
                return False

            preprocess_display = self.config.get("preprocess_mode", "默认")
            preprocess_mode = PREPROCESS_MODE_MAP.get(preprocess_display, "normal")

            # BUG-02 修复：使用 recognize_with_boxes 一次调用同时获取文本和位置
            ocr_manager = OCRManager()
            boxes_result = ocr_manager.recognize_with_boxes(
                screenshot, preprocess_mode=preprocess_mode
            )

            if not boxes_result:
                self._log_condition_result(False, "未识别到文本")
                return False

            # 从 boxes_result 提取全部文本
            all_text = "\n".join(text for text, _, _ in boxes_result if text)

            extract_mode_display = self.config.get("extract_mode", "全部")
            extract_mode = EXTRACT_MODE_MAP.get(extract_mode_display, "all")
            keywords = self.config.get("keywords", "")

            if extract_mode == "all":
                extracted_text = all_text
            else:
                extracted_text = self._extract_keywords_text(all_text, keywords)

            output_key = self.config.get("output_key", "last_extracted_text")
            context.blackboard.set(output_key, extracted_text)

            if self.config.get_bool("save_all_text", False):
                all_text_key = self.config.get("all_text_key", "all_ocr_text")
                context.blackboard.set(all_text_key, all_text)

            save_position = self.config.get_bool("save_position", True)
            if save_position:
                region = self._get_effective_region(context)
                region_left = region[0] if region else 0
                region_top = region[1] if region else 0

                # 从已有的 boxes_result 中查找关键词位置（不再重复调用 OCR）
                ocr_position = self._find_keyword_position(
                    boxes_result, keywords if extract_mode == "keywords" else None,
                    region_left, region_top
                )
                if ocr_position:
                    try:
                        from config.settings_manager import get_default_position_key
                        default_position_key = get_default_position_key()
                    except ImportError:
                        default_position_key = "last_detection_position"
                    position_key = self.config.get("position_key", "") or default_position_key
                    context.blackboard.set(position_key, ocr_position)
                    context.blackboard.set("last_detection_x", ocr_position[0])
                    context.blackboard.set("last_detection_y", ocr_position[1])
                else:
                    # 回退：使用区域中心
                    if region:
                        center_x = (region[0] + region[2]) // 2
                        center_y = (region[1] + region[3]) // 2
                        try:
                            from config.settings_manager import get_default_position_key
                            default_position_key = get_default_position_key()
                        except ImportError:
                            default_position_key = "last_detection_position"
                        position_key = self.config.get("position_key", "") or default_position_key
                        context.blackboard.set(position_key, (center_x, center_y))
                        # BUG-01 修复：回退路径补写 last_detection_x/y
                        context.blackboard.set("last_detection_x", center_x)
                        context.blackboard.set("last_detection_y", center_y)

            if extracted_text:
                self._log_condition_result(True)
                LogManager.instance().log_info(
                    node_type="文本提取节点",
                    node_name=self.name,
                    message=f"提取文本: {extracted_text[:50]}..."
                )
                return True
            else:
                self._log_condition_result(False, "未提取到匹配的文本")
                return False

        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"TextExtractNode '{self.name}'")
            self._log_condition_result(False, "执行异常，详情见终端日志")
            return False

    def _extract_keywords_text(self, all_text: str, keywords: str) -> str:
        lines = all_text.split('\n')
        matched_lines = []
        keywords_lower = keywords.lower()
        for line in lines:
            line = line.strip()
            if line and keywords_lower in line.lower():
                matched_lines.append(line)
        return '\n'.join(matched_lines)

    def _find_keyword_position(self, boxes_result, keywords: str = None,
                               region_left: int = 0, region_top: int = 0) -> Optional[Tuple[int, int]]:
        """从已有的 OCR boxes 结果中查找关键词位置

        Args:
            boxes_result: recognize_with_boxes 的返回结果 [(text, box, confidence), ...]
            keywords: 关键词（为None时使用第一个文本框位置）
            region_left: 区域左偏移
            region_top: 区域上偏移

        Returns:
            关键词中心位置 (x, y)，失败返回 None
        """
        try:
            # 如果有关键词，查找关键词所在文本框
            if keywords:
                keywords_lower = keywords.lower()
                for text, box, _ in boxes_result:
                    if keywords_lower in text.lower():
                        if box is not None:
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            center_x = int(sum(xs) / len(xs)) + region_left
                            center_y = int(sum(ys) / len(ys)) + region_top
                            return (center_x, center_y)

            # 无关键词或未找到，使用第一个文本框位置
            if boxes_result:
                _, box, _ = boxes_result[0]
                if box is not None:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    center_x = int(sum(xs) / len(xs)) + region_left
                    center_y = int(sum(ys) / len(ys)) + region_top
                    return (center_x, center_y)
        except Exception:
            pass
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextExtractNode":
        config = NodeConfig.from_dict(data.get("config", {}))
        node = cls(node_id=data.get("id"), config=config)
        return node
