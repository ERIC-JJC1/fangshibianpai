# scripts/verify_feeders_by_switch.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pandapower.networks as pn
import pandapower.topology as top

from topology.tie_map import inter_feeder_switch_map
from topology.tie_registry import annotate_tie_switches


def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def _build_net():
    # 和你 verify_tie_map 保持一致的网络构造参数（避免“同名不同网”的坑）
    net = pn.mv_oberrhein(
        scenario="generation",
        cosphi_load=0.98,
        cosphi_pv=1.0,
        include_substations=False,
        separation_by_sub=False,
    )
    return net


def _nx_graph(net, respect_switches: bool):
    # pandapower 官方拓扑建图：respect_switches=True 才是“按开关断开”的电气拓扑
    G = top.create_nxgraph(net, respect_switches=respect_switches)
    return G


def _component_index_map(G) -> Dict[int, int]:
    """
    返回: bus_id -> component_id
    component_id 按连通块大小从大到小排序后编号 0,1,2...
    """
    comps = list(top.connected_components(G))
    comps = sorted(comps, key=lambda s: len(s), reverse=True)
    bus2comp = {}
    for cid, comp in enumerate(comps):
        for b in comp:
            bus2comp[int(b)] = cid
    return bus2comp


def _print_components(G, title: str, top_k: int = 10):
    comps = list(top.connected_components(G))
    comps = sorted(comps, key=lambda s: len(s), reverse=True)
    sizes = [len(c) for c in comps]

    print(f"\n=== {title} ===")
    print(f"Connected components: {len(comps)}")
    print(f"Top {min(top_k, len(sizes))} component sizes: {sizes[:top_k]}")

    # 打印前几个连通块的 bus 样例（留痕）
    for i, comp in enumerate(comps[:min(4, len(comps))]):
        sample = sorted(list(comp))[:20]
        print(f"  - Comp {i}: size={len(comp)}, sample_buses={sample}")


def _extract_tie_pairs_from_map(tie_map: Dict) -> List[Tuple[int, int, int]]:
    """
    从 inter_feeder_switch_map 提取 tie pairs：
    返回列表 [(tie_id, sw_a, sw_b), ...]
    tie_id 就用 line 或者顺序编号都行；这里用 "line" 更好追溯
    """
    seen = set()
    pairs = []
    for feeder_id, links in tie_map.items():
        for link in links:
            sw1, sw2 = link["switch_pair"]
            line_id = int(link.get("line", -1))
            key = tuple(sorted((sw1, sw2)) + [line_id])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((line_id, int(sw1), int(sw2)))
    return sorted(pairs, key=lambda x: x[0])


def main():
    _ensure_logs_dir()
    net = _build_net()

    # 先把 tie 标注到 net.switch 上（后面打印对照会很方便）
    annotate_tie_switches(net, inter_feeder_switch_map)

    # 两种图：respect_switches False/True
    G_raw = _nx_graph(net, respect_switches=False)
    G_sw  = _nx_graph(net, respect_switches=True)

    print(f"[Graph raw] nodes={G_raw.number_of_nodes()}, edges={G_raw.number_of_edges()}")
    print(f"[Graph sw ] nodes={G_sw.number_of_nodes()}, edges={G_sw.number_of_edges()}")

    _print_components(G_raw, "Not respecting switches (pure topology)")
    _print_components(G_sw,  "Respecting switches (electrical topology)")

    bus2comp_raw = _component_index_map(G_raw)
    bus2comp_sw = _component_index_map(G_sw)

    tie_pairs = _extract_tie_pairs_from_map(inter_feeder_switch_map)

    # 逐个 tie 检查：两端 bus 在 respect_switches=True 的图里是否属于不同连通块？
    print("\n=== Tie pair cross-component check (respect_switches=True) ===")
    rows = []
    for tie_id, sw_a, sw_b in tie_pairs:
        if sw_a not in net.switch.index or sw_b not in net.switch.index:
            rows.append({
                "tie_id": tie_id, "sw_a": sw_a, "sw_b": sw_b,
                "ok": False, "reason": "switch id not in net.switch",
            })
            continue

        bus_a = int(net.switch.at[sw_a, "bus"])
        bus_b = int(net.switch.at[sw_b, "bus"])
        comp_a = bus2comp_sw.get(bus_a, -1)
        comp_b = bus2comp_sw.get(bus_b, -1)
        closed_a = bool(net.switch.at[sw_a, "closed"])
        closed_b = bool(net.switch.at[sw_b, "closed"])

        ok = (comp_a != -1 and comp_b != -1 and comp_a != comp_b)
        rows.append({
            "tie_id": tie_id,
            "sw_a": sw_a, "bus_a": bus_a, "comp_a": comp_a, "closed_a": closed_a,
            "sw_b": sw_b, "bus_b": bus_b, "comp_b": comp_b, "closed_b": closed_b,
            "cross_component": ok,
        })

    df = pd.DataFrame(rows)
    # 输出摘要
    if "cross_component" in df.columns:
        print(df[["tie_id", "sw_a", "bus_a", "comp_a", "closed_a",
                  "sw_b", "bus_b", "comp_b", "closed_b", "cross_component"]].to_string(index=False))
        print("\nSummary:")
        print(df["cross_component"].value_counts(dropna=False).to_string())
    else:
        print(df.to_string(index=False))

    # 落盘：作为“官方口径馈线分段证据”的结构化文件
    out_csv = os.path.join("logs", "feeders_by_switch_stage2_5.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")


if __name__ == "__main__":
    main()
