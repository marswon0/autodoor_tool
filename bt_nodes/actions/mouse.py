from bt_core.nodes import ActionNode, NodeStatus
from bt_core.config import NodeConfig
from typing import Dict, Any, Tuple, Optional
import random
import time
from bt_utils.log_manager import LogManager
from bt_utils.helpers import get_random_duration, get_random_interval


def _get_default_position_key() -> str:
    try:
        from config.settings_manager import get_default_position_key
        return get_default_position_key()
    except ImportError:
        return "last_detection_position"


def _ensure_tuple_position(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    return value


# 随机点击范围上限（检测目标尺寸的百分比），保证点击不超出检测图像
CLICK_RANDOM_PERCENT_MAX = 65


def get_position_mode(config, mode_key: str = "position_mode",
                      flag_key: str = "use_blackboard", var_key: str = "position_key") -> str:
    """位置来源：last_detection（最近检测点）/ fixed（固定坐标）/ variable（黑板变量）

    兼容旧节点（未保存 mode_key）：勾选旧的"使用最近检测点"且未填变量名 → 最近检测点；
    勾选且填了变量名 → 黑板变量；未勾选 → 固定坐标
    鼠标移动节点的终点使用 end_position_mode / use_blackboard_end / position_key_end
    """
    mode = config.get(mode_key, None)
    if mode:
        return mode
    if config.get_bool(flag_key, False):
        return "variable" if config.get(var_key, "") else "last_detection"
    return "fixed"


def _parse_click_count(value) -> Optional[int]:
    """有效点击次数：-1（无限）或 ≥1 的整数（兼容 "3"、3.0）；其他返回 None"""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if not isinstance(value, int):
        return None
    return value if value == -1 or value >= 1 else None


def _get_detection_position(context) -> Optional[Tuple[int, int]]:
    """最近检测点 + 检测节点的坐标偏移"""
    position = _ensure_tuple_position(context.blackboard.get(_get_default_position_key()))
    if position is None:
        return None
    offset = context.blackboard.get("last_detection_offset")
    if not (isinstance(offset, (list, tuple)) and len(offset) >= 2):
        offset = (0, 0)
    return (position[0] + int(offset[0]), position[1] + int(offset[1]))


class MouseClickNode(ActionNode):
    NODE_TYPE = "MouseClickNode"

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)
        self.button = self.config.get("button", "left")
        self.position: Optional[Tuple[int, int]] = _ensure_tuple_position(self.config.get("position", None))
        self.action = self.config.get("action", "press")
        self.duration = self.config.get_int("duration", 100)
        self.position_mode = get_position_mode(self.config)
        self.position_key = self.config.get("position_key", "")
        self.random_percent = self.config.get_int("random_percent", 0)
        self.click_count = self.config.get_int("click_count", 1)
        self.click_interval = self.config.get_int("click_interval", 100)
        self.duration_random = self.config.get_int("duration_random", 0)
        self.click_interval_random = self.config.get_int("click_interval_random", 0)
        self.x_float = self.config.get_int("x_float", 0)
        self.y_float = self.config.get_int("y_float", 0)
        self._current_click = 0
        self._resolved_click_count = 1
        self._last_click_time: Optional[float] = None
        self._actual_interval: Optional[int] = None
        self._actual_duration: Optional[int] = None
        self._abort_flag = False
        self._click_started = False
        self._button_pressed = False
        self._context = None
        self._no_size_logged = False

    def _resolve_click_count(self, context) -> int:
        """点击次数：固定值或黑板变量（每次开始点击时读取一次）

        变量名未设置、变量不存在或值不是有效次数时只点击一次，并记录日志
        """
        if self.config.get("click_count_mode", "fixed") == "variable":
            var_name = self.config.get("click_count_key", "")
            if not var_name:
                reason = "未设置点击次数变量名"
            elif not context.blackboard.exists(var_name):
                reason = f"黑板变量 '{var_name}' 不存在"
            else:
                value = context.blackboard.get(var_name)
                count = _parse_click_count(value)
                if count is not None:
                    return count
                reason = f"黑板变量 '{var_name}' 的值 {value!r} 不是有效点击次数"
        else:
            value = self.config.get("click_count", 1)
            count = _parse_click_count(value)
            if count is not None:
                return count
            reason = f"点击次数 {value!r} 无效"

        LogManager.instance().log_info(
            node_type="鼠标点击节点",
            node_name=self.name,
            message=f"{reason}，只点击一次"
        )
        return 1

    def _execute_action(self, context) -> NodeStatus:
        try:
            self._context = context
            click_position = self._get_position(context)
            if click_position is None and get_position_mode(self.config) != "fixed":
                self._release_button()
                LogManager.instance().log_failure(
                    node_type="鼠标点击节点",
                    node_name=self.name,
                    reason="黑板中没有可用的点击位置"
                )
                return NodeStatus.FAILURE

            if not self._click_started:
                self._click_started = True
                self._current_click = 0
                self._last_click_time = None
                self._resolved_click_count = self._resolve_click_count(context)

            if self._abort_flag or not context.check_running():
                self._release_button()
                LogManager.instance().log_aborted(
                    node_type="鼠标点击节点",
                    node_name=self.name
                )
                return NodeStatus.ABORTED

            if self._resolved_click_count == -1:
                return self._non_blocking_infinite_click(context, click_position)
            else:
                return self._non_blocking_finite_click(context, click_position)
            
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"MouseClickNode '{self.name}'")
            LogManager.instance().log_failure(
                node_type="鼠标点击节点",
                node_name=self.name,
                reason="执行异常，详情见终端日志"
            )
            return NodeStatus.FAILURE

    def _get_position(self, context) -> Optional[Tuple[int, int]]:
        mode = get_position_mode(self.config)
        if mode == "last_detection":
            return self._get_detection_click_position(context)
        if mode == "variable":
            position_key = self.config.get("position_key", "") or _get_default_position_key()
            return _ensure_tuple_position(context.blackboard.get(position_key))
        return _ensure_tuple_position(self.config.get("position", None))

    def _get_detection_click_position(self, context) -> Optional[Tuple[int, int]]:
        """最近检测点 + 检测节点的坐标偏移 + 检测尺寸内的随机点（每次点击重新随机）"""
        position = _get_detection_position(context)
        if position is None:
            return None

        percent = min(max(self.config.get_int("random_percent", 0), 0), CLICK_RANDOM_PERCENT_MAX)
        size = context.blackboard.get("last_detection_size")
        dx = dy = 0
        if percent > 0:
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                # 随机区域以检测中心为中心，边长为检测尺寸的 percent%
                half_w = size[0] * percent / 200
                half_h = size[1] * percent / 200
                dx = round(random.uniform(-half_w, half_w))
                dy = round(random.uniform(-half_h, half_h))
            elif not self._no_size_logged:
                self._no_size_logged = True
                LogManager.instance().log_info(
                    node_type="鼠标点击节点",
                    node_name=self.name,
                    message="最近检测无尺寸信息（如颜色/文字检测），随机点击范围不生效，点击检测点本身"
                )

        return (position[0] + dx, position[1] + dy)

    def _get_float_range(self) -> Tuple[int, int]:
        """±px 随机范围；最近检测点模式改用百分比随机，不叠加 ±px"""
        if get_position_mode(self.config) == "last_detection":
            return (0, 0)
        return (self.config.get_int("x_float", 0), self.config.get_int("y_float", 0))

    def _non_blocking_finite_click(self, context, position: Optional[Tuple[int, int]]) -> NodeStatus:
        current_time = time.time() * 1000

        click_interval = self.config.get_int("click_interval", 100)
        click_interval_random = self.config.get_int("click_interval_random", 0)

        if self._actual_interval is None:
            self._actual_interval = get_random_interval(click_interval, click_interval_random)

        if self._last_click_time is not None and self._actual_interval > 0:
            elapsed = current_time - self._last_click_time
            if elapsed < self._actual_interval:
                return NodeStatus.RUNNING

        click_count = self._resolved_click_count
        if self._current_click < click_count:
            if self._abort_flag or not context.check_running():
                self._release_button()
                return NodeStatus.ABORTED

            button = self.config.get("button", "left")
            action = self.config.get("action", "press")
            duration = self.config.get_int("duration", 100)
            duration_random = self.config.get_int("duration_random", 0)

            if self._actual_duration is None:
                self._actual_duration = get_random_duration(duration, duration_random)
            x_float, y_float = self._get_float_range()
            context.execute_mouse_click(button, position, action, self._actual_duration,
                                        x_float=x_float, y_float=y_float)

            if action == "down":
                self._button_pressed = True

            self._current_click += 1
            self._last_click_time = time.time() * 1000
            self._actual_interval = None
            self._actual_duration = None

            if self._current_click < click_count:
                return NodeStatus.RUNNING
        
        self._reset_click_state()
        LogManager.instance().log_success(
            node_type="鼠标点击节点",
            node_name=self.name
        )
        return NodeStatus.SUCCESS

    def _non_blocking_infinite_click(self, context, position: Optional[Tuple[int, int]]) -> NodeStatus:
        current_time = time.time() * 1000

        click_interval = self.config.get_int("click_interval", 100)
        click_interval_random = self.config.get_int("click_interval_random", 0)

        if self._actual_interval is None:
            self._actual_interval = get_random_interval(click_interval, click_interval_random)

        if self._last_click_time is not None and self._actual_interval > 0:
            elapsed = current_time - self._last_click_time
            if elapsed < self._actual_interval:
                return NodeStatus.RUNNING

        if self._abort_flag or not context.check_running():
            self._release_button()
            LogManager.instance().log_aborted(
                node_type="鼠标点击节点",
                node_name=self.name
            )
            return NodeStatus.ABORTED

        button = self.config.get("button", "left")
        action = self.config.get("action", "press")
        duration = self.config.get_int("duration", 100)
        duration_random = self.config.get_int("duration_random", 0)

        if self._actual_duration is None:
            self._actual_duration = get_random_duration(duration, duration_random)
        x_float, y_float = self._get_float_range()
        context.execute_mouse_click(button, position, action, self._actual_duration,
                                    x_float=x_float, y_float=y_float)

        if action == "down":
            self._button_pressed = True
        
        self._current_click += 1
        self._last_click_time = time.time() * 1000
        self._actual_interval = None
        self._actual_duration = None
        
        return NodeStatus.RUNNING

    def _release_button(self) -> None:
        if self._button_pressed and self._context:
            try:
                button = self.config.get("button", "left")
                self._context.execute_mouse_click(button, None, "up", 0)
            except Exception:
                pass
        self._reset_click_state()

    def _reset_click_state(self) -> None:
        self._current_click = 0
        self._last_click_time = None
        self._actual_interval = None
        self._actual_duration = None
        self._abort_flag = False
        self._click_started = False
        self._button_pressed = False

    def abort(self, context) -> None:
        self._abort_flag = True
        self._release_button()
        super().abort(context)

    def reset(self, reset_counters: bool = True) -> None:
        self._release_button()
        self._context = None
        self._no_size_logged = False
        super().reset(reset_counters=reset_counters)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MouseClickNode":
        config = NodeConfig.from_dict(data.get("config", {}))
        node = cls(node_id=data.get("id"), config=config)
        node.button = config.get("button", "left")
        node.position = _ensure_tuple_position(config.get("position", None))
        node.action = config.get("action", "press")
        node.duration = config.get_int("duration", 100)
        node.position_mode = get_position_mode(config)
        node.position_key = config.get("position_key", "")
        node.random_percent = config.get_int("random_percent", 0)
        node.click_count = config.get_int("click_count", 1)
        node.click_interval = config.get_int("click_interval", 100)
        node.duration_random = config.get_int("duration_random", 0)
        node.click_interval_random = config.get_int("click_interval_random", 0)
        node.x_float = config.get_int("x_float", 0)
        node.y_float = config.get_int("y_float", 0)
        return node


class MouseMoveNode(ActionNode):
    NODE_TYPE = "MouseMoveNode"

    def __init__(self, node_id: str = None, config: NodeConfig = None):
        super().__init__(node_id, config)
        self.position: Tuple[int, int] = _ensure_tuple_position(self.config.get("position", (0, 0))) or (0, 0)
        self.position_mode = get_position_mode(self.config)
        self.position_key = self.config.get("position_key", "")
        self.move_type = self.config.get("move_type", "移动")
        self.drag_button = self.config.get("drag_button", "left")
        self.end_position: Optional[Tuple[int, int]] = _ensure_tuple_position(self.config.get("end_position", None))
        self.relative = self._get_end_mode() == "relative"
        offset = self.config.get("offset", None)
        if offset is not None:
            if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                self.offset = (int(offset[0]), int(offset[1]))
            else:
                self.offset = (0, 0)
        else:
            self.offset = (0, 0)
        self.end_position_mode = self._get_end_mode()
        self.position_key_end = self.config.get("position_key_end", "")
        self.move_duration = self.config.get_int("move_duration", 0)
        self.move_duration_random = self.config.get_int("move_duration_random", 0)
        self.drag_duration = self.config.get_int("drag_duration", 0)
        self.drag_duration_random = self.config.get_int("drag_duration_random", 0)
        self.x_float = self.config.get_int("x_float", 0)
        self.y_float = self.config.get_int("y_float", 0)

    def _execute_action(self, context) -> NodeStatus:
        try:
            start_pos = self._get_start_position(context)

            if not start_pos:
                LogManager.instance().log_failure(
                    node_type="鼠标移动节点",
                    node_name=self.name,
                    reason="未指定起点位置" if get_position_mode(self.config) == "fixed" else "黑板中没有可用的起点位置"
                )
                return NodeStatus.FAILURE

            end_pos = self._get_end_position(context, start_pos)
            if end_pos is None and self._get_end_mode() != "fixed":
                LogManager.instance().log_failure(
                    node_type="鼠标移动节点",
                    node_name=self.name,
                    reason="黑板中没有可用的终点位置"
                )
                return NodeStatus.FAILURE

            move_type = self.config.get("move_type", "移动")
            if move_type == "拖拽":
                return self._execute_drag(context, start_pos, end_pos)
            else:
                return self._execute_move(context, start_pos, end_pos)
                
        except Exception as e:
            from bt_utils.exception_handler import log_exception
            log_exception(e, f"MouseMoveNode '{self.name}'")
            LogManager.instance().log_failure(
                node_type="鼠标移动节点",
                node_name=self.name,
                reason="执行异常，详情见终端日志"
            )
            try:
                context.execute_mouse_click(self.config.get("drag_button", "left"), None, "up", 0)
            except Exception:
                pass
            return NodeStatus.FAILURE

    def _get_start_position(self, context) -> Optional[Tuple[int, int]]:
        mode = get_position_mode(self.config)
        if mode == "last_detection":
            return _get_detection_position(context)
        if mode == "variable":
            position_key = self.config.get("position_key", "") or _get_default_position_key()
            return _ensure_tuple_position(context.blackboard.get(position_key))
        return _ensure_tuple_position(self.config.get("position", (0, 0)))

    def _get_end_mode(self) -> str:
        """终点位置来源：在起点三种来源之外多一种 relative（起点+偏移）

        兼容旧节点（未保存 end_position_mode）：勾选了旧的"增量移动"→ 起点+偏移
        """
        if not self.config.get("end_position_mode", None) and self.config.get_bool("relative", False):
            return "relative"
        return get_position_mode(self.config, "end_position_mode", "use_blackboard_end", "position_key_end")

    def _get_end_position(self, context, start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        mode = self._get_end_mode()
        if mode == "relative":
            offset = self.config.get("offset", None)
            if offset is not None and isinstance(offset, (list, tuple)) and len(offset) >= 2:
                return (start_pos[0] + int(offset[0]), start_pos[1] + int(offset[1]))
            return start_pos
        if mode == "last_detection":
            return _get_detection_position(context)
        if mode == "variable":
            position_key_end = self.config.get("position_key_end", "") or _get_default_position_key()
            return _ensure_tuple_position(context.blackboard.get(position_key_end))
        return _ensure_tuple_position(self.config.get("end_position", None))

    def _smoothstep(self, t: float) -> float:
        return t * t * (3 - 2 * t)

    def _execute_move(self, context, start_pos: Tuple[int, int], end_pos: Optional[Tuple[int, int]]) -> NodeStatus:
        if not end_pos:
            context.execute_mouse_move(start_pos, relative=False,
                                       x_float=self.x_float, y_float=self.y_float)
            LogManager.instance().log_success(
                node_type="鼠标移动节点",
                node_name=self.name
            )
            return NodeStatus.SUCCESS

        move_duration = self.config.get_int("move_duration", 0)
        move_duration_random = self.config.get_int("move_duration_random", 0)
        actual_duration = get_random_duration(move_duration, move_duration_random)
        
        context.execute_mouse_move(start_pos, relative=False,
                                   x_float=self.x_float, y_float=self.y_float)
        time.sleep(0.01)
        
        if actual_duration > 0:
            total_duration_sec = actual_duration / 1000.0
            steps = max(10, actual_duration // 50)
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            start_time = time.time()
            
            for i in range(steps):
                if not context.check_running():
                    return NodeStatus.ABORTED
                
                elapsed = time.time() - start_time
                progress = min(elapsed / total_duration_sec, 1.0)
                
                t = self._smoothstep(progress)
                current_x = int(start_pos[0] + dx * t)
                current_y = int(start_pos[1] + dy * t)
                context.execute_mouse_move((current_x, current_y), relative=False,
                                           x_float=self.x_float, y_float=self.y_float)
                
                if progress >= 1.0:
                    break
                
                time.sleep(total_duration_sec / steps)
        else:
            context.execute_mouse_move(end_pos, relative=False,
                                       x_float=self.x_float, y_float=self.y_float)
        
        LogManager.instance().log_success(
            node_type="鼠标移动节点",
            node_name=self.name
        )
        return NodeStatus.SUCCESS

    def _execute_drag(self, context, start_pos: Tuple[int, int], end_pos: Optional[Tuple[int, int]]) -> NodeStatus:
        if not end_pos:
            LogManager.instance().log_failure(
                node_type="鼠标移动节点",
                node_name=self.name,
                reason="未指定拖拽终点"
            )
            return NodeStatus.FAILURE

        move_duration = self.config.get_int("move_duration", 0)
        move_duration_random = self.config.get_int("move_duration_random", 0)
        drag_duration = self.config.get_int("drag_duration", 0)
        drag_duration_random = self.config.get_int("drag_duration_random", 0)
        drag_button = self.config.get("drag_button", "left")

        actual_duration = get_random_duration(
            move_duration if move_duration > 0 else drag_duration,
            move_duration_random if move_duration_random > 0 else drag_duration_random
        )

        context.execute_mouse_move(start_pos, relative=False,
                                   x_float=self.x_float, y_float=self.y_float)
        time.sleep(0.02)

        context.execute_mouse_click(drag_button, start_pos, "down", 0)
        time.sleep(0.02)
        
        if actual_duration > 0:
            total_duration_sec = actual_duration / 1000.0
            steps = max(10, actual_duration // 50)
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            start_time = time.time()
            
            for i in range(steps):
                if not context.check_running():
                    context.execute_mouse_click(drag_button, end_pos, "up", 0)
                    return NodeStatus.ABORTED
                
                elapsed = time.time() - start_time
                progress = min(elapsed / total_duration_sec, 1.0)
                
                t = self._smoothstep(progress)
                current_x = int(start_pos[0] + dx * t)
                current_y = int(start_pos[1] + dy * t)
                context.execute_mouse_move((current_x, current_y), relative=False,
                                           x_float=self.x_float, y_float=self.y_float)
                
                if progress >= 1.0:
                    break
                
                time.sleep(total_duration_sec / steps)
        else:
            context.execute_mouse_move(end_pos, relative=False,
                                       x_float=self.x_float, y_float=self.y_float)
        
        time.sleep(0.02)
        context.execute_mouse_click(drag_button, end_pos, "up", 0)

        LogManager.instance().log_success(
            node_type="鼠标移动节点",
            node_name=self.name
        )
        return NodeStatus.SUCCESS

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MouseMoveNode":
        config = NodeConfig.from_dict(data.get("config", {}))
        node = cls(node_id=data.get("id"), config=config)
        node.position = _ensure_tuple_position(config.get("position", (0, 0))) or (0, 0)
        node.position_mode = get_position_mode(config)
        node.position_key = config.get("position_key", "")
        node.move_type = config.get("move_type", "移动")
        node.drag_button = config.get("drag_button", "left")
        node.end_position = _ensure_tuple_position(config.get("end_position", None))
        node.relative = node._get_end_mode() == "relative"
        offset = config.get("offset", None)
        if offset is not None:
            if isinstance(offset, (list, tuple)) and len(offset) >= 2:
                node.offset = (int(offset[0]), int(offset[1]))
            else:
                node.offset = (0, 0)
        else:
            node.offset = (0, 0)
        node.end_position_mode = node._get_end_mode()
        node.position_key_end = config.get("position_key_end", "")
        node.move_duration = config.get_int("move_duration", 0)
        node.move_duration_random = config.get_int("move_duration_random", 0)
        node.drag_duration = config.get_int("drag_duration", 0)
        node.drag_duration_random = config.get_int("drag_duration_random", 0)
        node.x_float = config.get_int("x_float", 0)
        node.y_float = config.get_int("y_float", 0)
        return node
