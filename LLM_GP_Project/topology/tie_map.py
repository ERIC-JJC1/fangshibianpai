# -*- coding: utf-8 -*-
"""
topology/tie_map.py

作用（给小白看的）：
- 把“mv_oberrhein 默认网络里哪些线路是断点/联络点（section points）”写成一份固定清单
- 这份清单是 Stage 2 拓扑核验的“证据基线”
- 后续你要增强网络（新增 tie、改默认开关状态），也在这里改，并且有日志可追溯

IMPORTANT:
- OBERRHEIN_SECTION_POINTS：以 pandapower 默认 net.switch 状态为准（你日志里那 6 条线）
- inter_feeder_switch_map：保留一个兼容名字，避免你其他脚本 import 报错
"""

from __future__ import annotations

# 这 6 条就是你 logs/inspect_section_points_stage2_6.txt 里打印出来的结果
# 每条 line 上有 2 个开关：一个闭合(True)，一个断开(False)
OBERRHEIN_SECTION_POINTS = {
    8:   {"switch_pair": (13, 14),  "bus_pair": (167, 129), "default_closed": (True,  False)},
    23:  {"switch_pair": (33, 34),  "bus_pair": (195, 132), "default_closed": (True,  False)},
    31:  {"switch_pair": (47, 48),  "bus_pair": (31,  190), "default_closed": (True,  False)},
    66:  {"switch_pair": (106, 107),"bus_pair": (54,  147), "default_closed": (True,  False)},
    88:  {"switch_pair": (143, 144),"bus_pair": (236, 223), "default_closed": (True,  False)},
    188: {"switch_pair": (310, 311),"bus_pair": (35,   45), "default_closed": (True,  False)},
}

# 兼容旧代码：有些脚本可能还在 from topology.tie_map import inter_feeder_switch_map
# 这里先给一个空的占位，避免 ImportError。
# 真正“按4条馈线组织的 inter_feeder_switch_map”建议后面 Stage 3 再从 feeder_map 自动生成。
inter_feeder_switch_map = {}
