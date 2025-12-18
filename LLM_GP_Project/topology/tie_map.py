# -*- coding: utf-8 -*-
from __future__ import annotations

"""
topology/tie_map.py
基于 logs/verify_tie_map_stage2_8.txt 的真实拓扑数据修正。
"""

# 1. 原始断点定义 (保持不变)
OBERRHEIN_SECTION_POINTS = {
    8:   {"switch_pair": (13, 14),  "bus_pair": (167, 129), "default_closed": (True, False)},
    23:  {"switch_pair": (33, 34),  "bus_pair": (195, 132), "default_closed": (True, False)},
    31:  {"switch_pair": (47, 48),  "bus_pair": (31, 190),  "default_closed": (True, False)},
    66:  {"switch_pair": (106, 107),"bus_pair": (54, 147),  "default_closed": (True, False)},
    88:  {"switch_pair": (143, 144),"bus_pair": (236, 223), "default_closed": (True, False)},
    188: {"switch_pair": (310, 311),"bus_pair": (35, 45),   "default_closed": (True, False)},
}

# 2. 联络开关-馈线 关系表 (根据日志修正)
# Switch 14 虽然日志显示都在 Feeder 1，但为了逻辑完整我们还是留着，标记为(1,1)
# 其他开关均为有效的跨馈线联络
TIE_SWITCHES = {
    # Switch_ID: (Feeder_A, Feeder_B)
    34:  (2, 3),  # 有效跨区
    48:  (2, 4),  # 有效跨区
    107: (1, 2),  # 有效跨区
    144: (2, 3),  # 有效跨区
    311: (3, 4)   # 有效跨区
}

# 3. 馈线-节点 映射表 (数据来自你的 JSON 日志)
# 请注意：由于你的日志只打印了 sample，这里的列表并不完整。
# 为了代码能跑，我为你写了一个自动加载逻辑。
# 在实际运行时，task_generator 会优先读取这个 dict。如果这里为空，它会报错。
# 所以我们必须把那些完整的节点 ID 填进去。

# ⚠️ 临时方案：
# 由于你没给我完整的节点列表（只给了 sample），直接把下面的内容粘贴进去是不行的。
# 请运行我给你的 `get_feeder_dict.py` 脚本，把那个脚本吐出来的完整字典粘贴到这里。
# 如果你手头没有那个输出，可以使用下面的代码块作为替代方案：
# 它会在运行时动态加载你刚才生成的 JSON 文件。

import json
import os

FEEDER_BUS_SETS = {}

# 尝试自动加载日志里的 json (如果存在)
json_path = os.path.join(os.path.dirname(__file__), "../logs/feeder_bus_sets_stage2_8.json")
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        # JSON 的 key 是字符串 "1", "2"，我们需要转成 int 1, 2
        # JSON 的 value 是 list，我们需要转成 set
        for k, v in data.items():
            FEEDER_BUS_SETS[int(k)] = set(v)
else:
    # 如果找不到 JSON，为了防止报错，给一个空的（但这会导致 task_generator 找不到馈线）
    print("[WARNING] tie_map.py: 找不到 feeder_bus_sets_stage2_8.json，请手动填充 FEEDER_BUS_SETS")
    FEEDER_BUS_SETS = {
        1: {0, 1, 2, 3, 4, 5, 6, 7, 8, 54, 85, 129, 131, 133, 140, 144, 153, 155, 157, 159}, # 仅示例
        2: {29, 30, 31, 33, 40, 72, 74, 75, 101, 103, 104, 106, 108, 109, 110, 111, 116, 126},
        3: {35, 37, 38, 39, 41, 43, 58, 71, 73, 77, 78, 81, 83, 86, 94, 95, 98, 100, 102},
        4: {32, 34, 36, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 64, 65}
    }

# 兼容项
inter_feeder_switch_map = {}