# -*- coding: utf-8 -*-
"""
topology/tie_map.py

统一的“联络开关/分段点”定义出口。

对外提供两个接口：
1) inter_feeder_switch_map: 你原来环境用的结构（按 feeder_id 分组）
2) TIE_DEFS: 更通用的列表结构（每个 tie 一个 dict）

注意：verify_tie_map.py 目前 import 的是 inter_feeder_switch_map，
所以这里必须存在这个变量名。
"""

from __future__ import annotations

# -----------------------------
# 1) 你的原始结构：按馈线编号组织
# -----------------------------
# 你可以把你现在用的那份字典原样粘过来（我这里给你放一个“占位示例”）
# !!! 请把下面内容替换成你真实的 inter_feeder_switch_map !!!
inter_feeder_switch_map = {
    1: [
        {"with_feeder": 2, "switch_pair": [106, 107], "line": 66, "bus1": 54, "bus2": 147},
    ],
    2: [
        {"with_feeder": 1, "switch_pair": [106, 107], "line": 66, "bus1": 147, "bus2": 54},
        {"with_feeder": 3, "switch_pair": [33, 34], "line": 23, "bus1": 195, "bus2": 132},
        {"with_feeder": 3, "switch_pair": [143, 144], "line": 88, "bus1": 236, "bus2": 223},
        {"with_feeder": 4, "switch_pair": [47, 48], "line": 31, "bus1": 31, "bus2": 190},
    ],
    3: [
        {"with_feeder": 2, "switch_pair": [33, 34], "line": 23, "bus1": 132, "bus2": 195},
        {"with_feeder": 2, "switch_pair": [143, 144], "line": 88, "bus1": 223, "bus2": 236},
        {"with_feeder": 4, "switch_pair": [264, 265], "line": 162, "bus1": 39, "bus2": 80},
        {"with_feeder": 4, "switch_pair": [310, 311], "line": 188, "bus1": 35, "bus2": 45},
    ],
    4: [
        {"with_feeder": 2, "switch_pair": [47, 48], "line": 31, "bus1": 190, "bus2": 31},
        {"with_feeder": 3, "switch_pair": [264, 265], "line": 162, "bus1": 80, "bus2": 39},
        {"with_feeder": 3, "switch_pair": [310, 311], "line": 188, "bus1": 45, "bus2": 35},
    ],
}

# -----------------------------
# 2) 通用结构：每个 tie 一条记录
# -----------------------------
# 从 inter_feeder_switch_map 自动汇总，避免重复写两份
TIE_DEFS = []
_seen = set()
for feeder_id, links in inter_feeder_switch_map.items():
    for link in links:
        # 用 line_id + switch_pair 做唯一键（避免同一条 tie 在 feeder A、B 各写一遍）
        key = (int(link["line"]), tuple(sorted(map(int, link["switch_pair"]))))
        if key in _seen:
            continue
        _seen.add(key)
        TIE_DEFS.append(
            {
                "line": int(link["line"]),
                "switch_pair": list(map(int, link["switch_pair"])),
                "bus1": int(link.get("bus1")),
                "bus2": int(link.get("bus2")),
            }
        )

# 按 line 排序一下，方便读
TIE_DEFS = sorted(TIE_DEFS, key=lambda d: d["line"])
