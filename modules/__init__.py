"""
模块注册器 - 将自定义模块注入 ultralytics

方法: 通过 exec 重写 parse_model，扩展 base_modules 和 repeat_modules
使自定义模块获得与原生模块相同的参数自动处理

用法:
    from modules import register_modules, make_model
    register_modules()
    model = make_model('cfgs/yolov8-DBB-DCNV4.yaml')
"""

import inspect
import sys
import os

from .dbb import DBB, DBBottleNeck, DeepDBBottleNeck, WDBBottleNeck
from .dcnv import DCNv3_pytorch, DCNv4_pytorch, get_dcnv3, get_dcnv4, HAS_DCNV3_CUDA, HAS_DCNV4_CUDA
from .c2f_variants import (
    C2f_DBB, C2f_DeepDBB, C2f_WDBB,
    C2f_DCNV3, C2f_DCNV4,
    DCNV4_Detect,
)

__all__ = [
    'C2f_DBB', 'C2f_DeepDBB', 'C2f_WDBB',
    'C2f_DCNV3', 'C2f_DCNV4',
    'DCNV4_Detect',
    'register_modules', 'make_model',
]

# 自定义模块列表
CUSTOM_CLASSES = {
    'C2f_DBB': C2f_DBB,
    'C2f_DeepDBB': C2f_DeepDBB,
    'C2f_WDBB': C2f_WDBB,
    'C2f_DCNV3': C2f_DCNV3,
    'C2f_DCNV4': C2f_DCNV4,
    'DCNV4_Detect': DCNV4_Detect,
}
CUSTOM_NAMES = set(CUSTOM_CLASSES.keys())

# 哪些是 repeat 模块
REPEAT_NAMES = {'C2f_DBB', 'C2f_DeepDBB', 'C2f_WDBB', 'C2f_DCNV3', 'C2f_DCNV4'}


def make_model(yaml_path, scale='n'):
    """构建 YOLO 模型 (自定义模块兼容版)"""
    register_modules()
    from ultralytics import YOLO
    return YOLO(yaml_path, task='detect')


def register_modules():
    """
    注册自定义模块到 ultralytics
    
    策略:
      1. 将模块注入 ultralytics.nn.tasks globals (parse_model 通过 globals()[m] 查找)
      2. 重写 parse_model 函数, 扩展 base_modules 和 repeat_modules frozenset
    
    每次模块更新后只需调用一次。
    """
    import ultralytics.nn.tasks as tasks
    
    # ===== 1. 注入 globals =====
    for name, cls in CUSTOM_CLASSES.items():
        setattr(tasks, name, cls)
    
    # 也注入模块可能被其他地方导入的地方
    import ultralytics.nn.modules as ultra_mods
    for name, cls in CUSTOM_CLASSES.items():
        if not hasattr(ultra_mods, name):
            setattr(ultra_mods, name, cls)
    
    # ===== 2. 扩展 parse_model 的 base_modules/repeat_modules =====
    # 通过获取源码 → 修改 frozenset → exec 重新定义
    if hasattr(tasks, '_patched_parse_model'):
        return True  # 已打过补丁
    
    src = inspect.getsource(tasks.parse_model)
    
    # 构建自定义模块的 frozenset 插入代码
    custom_base_entries = '\n        '.join(f'{n},' for n in sorted(CUSTOM_NAMES))
    custom_repeat_entries = '\n        '.join(f'{n},' for n in sorted(REPEAT_NAMES))
    
    # 在 base_modules 的结尾插入自定义模块
    # 查找 frozenset 闭合 })前插入
    base_insertion = src.find('        }\n    )\n    repeat_modules')
    if base_insertion > 0:
        src = src[:base_insertion] + f'        # 自定义 DBB/DCNV 模块\n        {custom_base_entries}\n' + src[base_insertion:]
    
    # 在 repeat_modules 的结尾插入
    repeat_insertion = src.find('        }\n    )\n    for i, (f, n, m, args)')
    if repeat_insertion > 0:
        src = src[:repeat_insertion] + f'        # 自定义 DBB/DCNV 模块\n        {custom_repeat_entries}\n' + src[repeat_insertion:]
    
    # 准备 exec 需要的全局变量
    # 收集 parse_model 源码中引用的所有模块
    exec_globals = {}
    
    # 从原始函数获取 globals
    orig_globals = tasks.parse_model.__globals__
    
    # 复制所有必要的全局变量
    for key in orig_globals:
        exec_globals[key] = orig_globals[key]
    
    # 注入我们的自定义模块
    for name, cls in CUSTOM_CLASSES.items():
        exec_globals[name] = cls
    
    # 注入其他可能需要的引用
    exec_globals['__builtins__'] = __builtins__
    
    # 执行修改后的源码
    exec(src, exec_globals)
    
    # 提取重新定义的 parse_model
    new_parse_model = exec_globals['parse_model']
    
    # 替换到 tasks 模块
    tasks.parse_model = new_parse_model
    tasks._patched_parse_model = new_parse_model
    
    return True
