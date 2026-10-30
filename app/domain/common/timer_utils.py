import random, os, time
import uuid
from typing import Any, Dict, List, Sequence, Set, Iterator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

def is_image(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTS)


def get_all_dirs(root_dir: str, skip_hidden: bool = False) -> List[str]:
    """返回 root_dir 以及其所有子目录的列表；默认不跟随符号链接。"""
    all_dirs = [root_dir]
    stack = [root_dir]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        # 跳过隐藏目录（如 .git、.venv 等）
                        if skip_hidden and entry.name.startswith('.'):
                            continue
                        d = entry.path
                        all_dirs.append(d)
                        stack.append(d)
        except PermissionError:
            # 跳过权限不足的目录
            continue
    return all_dirs


def get_dir_all_images(directory: str, exts: Set[str]) -> List[str]:
    """
    获取指定目录下所有图像文件路径（不递归子目录）。
    使用 os.scandir() 提升性能。
    """
    image_paths: List[str] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                # 检查是否是文件，并且扩展名匹配
                if entry.is_file(follow_symlinks=False):
                    name_lower = entry.name.lower()
                    if any(name_lower.endswith(ext) for ext in exts):
                        image_paths.append(entry.path)
    except PermissionError:
        pass
    return image_paths


def get_all_image_paths(
    directory, 
    filter_str: List[str] = None,
    exclude_str: List[str] = None,
    image_exts: Set[str] = IMAGE_EXTS,
    ratio = 1.0, 
    seed = None,
    min_select: int = 0,
    ) -> List[str]:
    filter_str = filter_str or []
    exclude_str = exclude_str or []
    
    if seed is not None: 
        random.seed(seed)
    
    all_dirs = get_all_dirs(directory)
    
    total_pathList_dict = {}
    total_pathList = []
    for dir_path in all_dirs:
        collected = []
        image_pathList = get_dir_all_images(dir_path, image_exts)
        
        if not image_pathList:
            continue
        if filter_str:
            image_pathList = [p for p in image_pathList if any(s in p for s in filter_str)]
            if not image_pathList:
                continue
        # —— 排除过滤：剔除路径中含任一 exclude 关键词的
        if exclude_str:
            image_pathList = [p for p in image_pathList if not any(ex in p for ex in exclude_str)]
            if not image_pathList:
                continue
        collected.extend(image_pathList)
        if 0 < ratio < 1.0:
            num_to_select = max(min_select, int(len(collected) * ratio))
            num_to_select = min(num_to_select, len(collected))
            if num_to_select <= 0: 
                continue
            collected = random.sample(collected, num_to_select)
        total_pathList.extend(collected)
        total_pathList_dict[dir_path] = collected
    return total_pathList, total_pathList_dict



class timer_tool:
    def __init__(self) -> None:
        """
        初始化计时器工具
        设置计时起点和上一个标记点，并初始化一个空列表用于存储时间标记
        """
        self._start = time.perf_counter()  # 整个周期的起点
        self._last = self._start           # 上一个 mark 的时间点
        self._buf: List[str] = []          # 用于存储时间标记的列表

    def mark(self, tag: str) -> None:
        """
        添加一个时间标记点
        计算与上一个标记点的时间差，并将标记信息添加到缓冲区
        :param tag: 标记点的名称
        """
        now = time.perf_counter()
        self._buf.append(f"{tag} {(now - self._last):.3f}s   ")  # 添加标记信息，包含名称和耗时
        self._last = now                                        # 更新上一个标记点为当前时间

    def summary(self) -> str:

        """
        生成计时摘要
        计算总耗时并返回所有标记点信息
        :return: 包含所有标记点和总耗时的字符串
        """
        total = time.perf_counter() - self._start  # 计算总耗时
        return "".join(self._buf) + f"|| total {total:.3f}s"
    

def show_text(trace_id: str, message: str, image_name:str="", time_spend: str = ""):
    show_dict = {}
    show_dict["index"] = "roki_ai_food_identification"
    show_dict["traceId"] = trace_id or str(uuid.uuid4())
    show_dict['imageId'] = image_name
    show_dict["spendTime"] = time_spend
    show_dict["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    show_dict["message"] = message
    show_dict["level"] = "INFO"
    print(str(show_dict).replace("'", '"').replace("\\n", " ").replace("\\t", "  "))