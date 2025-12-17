# -*- coding: utf-8 -*-
"""
scripts/verify_tie_map.py

运行方式：
    python -m scripts.verify_tie_map | tee logs/verify_tie_map_stage2_7.txt
"""

from __future__ import annotations

import os
import pandapower.networks as pn

from topology.tie_registry import validate_inter_feeder_switch_map, annotate_tie_switches
from topology.tie_map import OBERRHEIN_SECTION_POINTS


def build_pairs_from_section_points(section_points: dict) -> list[dict]:
    pairs = []
    for line_id, info in section_points.items():
        sw_a, sw_b = info["switch_pair"]
        pairs.append({
            "line": int(line_id),                 # tie_registry 常用的字段名
            "switch_pair": [int(sw_a), int(sw_b)] # ✅ 关键：必须叫 switch_pair
        })
    return pairs



def main():
    os.makedirs("logs", exist_ok=True)

    net = pn.mv_oberrhein(
        scenario="generation",
        cosphi_load=0.98,
        cosphi_pv=1.0,
        include_substations=False,
        separation_by_sub=False,
    )

    pairs = build_pairs_from_section_points(OBERRHEIN_SECTION_POINTS)

    # ✅ 兼容 tie_registry：它期望 tie_map 是 dict，因此我们包一层
    tie_map_compat = {"SECTION_POINTS": pairs}

    issues_df = validate_inter_feeder_switch_map(net, tie_map_compat)

    print("\n=== tie_map 校验统计 ===")
    if len(issues_df) == 0:
        print("No issues. (0)")
    else:
        print(issues_df["level"].value_counts())

    annotate_tie_switches(net, tie_map_compat)

    print("\n=== 标注后的 tie switch（前20行） ===")
    cols = ["bus", "et", "element", "closed", "tie_id", "tie_peer_switch"]
    print(net.switch.loc[net.switch["tie_id"].notna(), cols].head(20))

    tie_sw = net.switch[net.switch["tie_id"].notna()]
    n_tie = len(tie_sw)
    n_open = int((tie_sw["closed"] == False).sum())
    n_closed = int((tie_sw["closed"] == True).sum())
    print(f"\nTie switches: {n_tie}, open(closed=False): {n_open}, closed: {n_closed}")

    out_csv = "logs/tie_map_issues_stage2.csv"
    issues_df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved issues csv: {out_csv}")

    if len(issues_df) > 0:
        print("\n=== Issues preview (top 30) ===")
        print(issues_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
