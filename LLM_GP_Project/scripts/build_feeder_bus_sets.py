# -*- coding: utf-8 -*-
"""
build_feeder_bus_sets.py

目标：
1) 从 pandapower 的 mv_oberrhein 网络里，构建“考虑开关状态的电气拓扑图”
2) 以多个 root bus（你定义的 4 条馈线源点）做多源 BFS，给每个 bus 分配 feeder_id
3) 输出 FEEDER_BUS_SETS（馈线->bus集合），并保存可追溯产物到 logs/
4) 可选：自动把 FEEDER_BUS_SETS 写回 topology/tie_map.py（--write-tie-map）

运行方式（推荐在 LLM_GP_Project 根目录）：
  python -m scripts.build_feeder_bus_sets | tee logs/build_feeder_bus_sets_stage2_8.txt

如果要自动写回 tie_map.py：
  python -m scripts.build_feeder_bus_sets --write-tie-map | tee logs/build_feeder_bus_sets_stage2_8.txt
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import pandapower.networks as pn


# =========================
# 你可以在这里固定“4条馈线的 root bus”
# （和你之前 traverse_feeder 那套一致）
# =========================
ROOT_BUSES: Dict[int, int] = {
    1: 319,
    2: 126,
    3: 58,
    4: 80,
}


def _line_is_electrically_connected(net, line_id: int) -> bool:
    """线路是否电气连通：line.in_service=True 且该线路上所有 line-switch 都是 closed=True"""
    if not bool(net.line.at[line_id, "in_service"]):
        return False
    sw = net.switch
    if sw is None or len(sw) == 0:
        return True
    line_sws = sw[(sw["et"] == "l") & (sw["element"] == line_id)]
    if len(line_sws) == 0:
        return True
    # 任何一个开关断开 => 该线路不连通
    return bool(line_sws["closed"].all())


def _trafo_is_electrically_connected(net, trafo_id: int) -> bool:
    """变压器是否电气连通：trafo.in_service=True 且该 trafo 上所有 trafo-switch 都是 closed=True"""
    if not bool(net.trafo.at[trafo_id, "in_service"]):
        return False
    sw = net.switch
    if sw is None or len(sw) == 0:
        return True
    trafo_sws = sw[(sw["et"] == "t") & (sw["element"] == trafo_id)]
    if len(trafo_sws) == 0:
        return True
    return bool(trafo_sws["closed"].all())


def build_electrical_adjacency(net) -> Dict[int, Set[int]]:
    """
    构建“电气拓扑”邻接表：考虑 switch.closed 状态
    - 线路：line + 线路开关
    - 变压器：trafo + 变压器开关（如果存在）
    """
    adj: Dict[int, Set[int]] = defaultdict(set)

    # lines
    for lid, row in net.line.iterrows():
        if not _line_is_electrically_connected(net, int(lid)):
            continue
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        adj[fb].add(tb)
        adj[tb].add(fb)

    # trafos
    if hasattr(net, "trafo") and net.trafo is not None and len(net.trafo) > 0:
        for tid, row in net.trafo.iterrows():
            if not _trafo_is_electrically_connected(net, int(tid)):
                continue
            hv = int(row["hv_bus"])
            lv = int(row["lv_bus"])
            adj[hv].add(lv)
            adj[lv].add(hv)

    # 兜底：保证所有 bus 都在字典里（哪怕孤岛）
    for b in net.bus.index:
        _ = adj[int(b)]

    return adj


def multi_source_bfs_partition(adj: Dict[int, Set[int]], roots: Dict[int, int]) -> Dict[int, int]:
    """
    多源 BFS：从多个 root 同时扩散，给每个 bus 分配 feeder_id
    规则：谁先到达就归谁（相当于按最短路距离划分），若同距则按 feeder_id 小者先入队（稳定）
    """
    owner: Dict[int, int] = {}
    q = deque()

    # feeder_id 排序保证可复现
    for feeder_id in sorted(roots.keys()):
        rb = int(roots[feeder_id])
        owner[rb] = feeder_id
        q.append(rb)

    while q:
        u = q.popleft()
        fid = owner[u]
        for v in adj[u]:
            if v not in owner:
                owner[v] = fid
                q.append(v)

    return owner


def invert_owner_to_sets(owner: Dict[int, int]) -> Dict[int, List[int]]:
    feeder_sets: Dict[int, List[int]] = defaultdict(list)
    for bus, fid in owner.items():
        feeder_sets[int(fid)].append(int(bus))
    for fid in feeder_sets:
        feeder_sets[fid] = sorted(feeder_sets[fid])
    return dict(feeder_sets)


def connected_components_from_adj(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    """仅用于打印调试：电气拓扑下的连通分量"""
    seen = set()
    comps = []
    for n in adj.keys():
        if n in seen:
            continue
        stack = [n]
        comp = set([n])
        seen.add(n)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def save_artifacts(log_dir: Path, feeder_bus_sets: Dict[int, List[int]], owner: Dict[int, int]):
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1) json
    json_path = log_dir / "feeder_bus_sets_stage2_8.json"
    json_path.write_text(json.dumps(feeder_bus_sets, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) csv: bus -> feeder
    df = pd.DataFrame({"bus": list(owner.keys()), "feeder_id": list(owner.values())}).sort_values(["feeder_id", "bus"])
    csv_path = log_dir / "feeder_by_bus_stage2_8.csv"
    df.to_csv(csv_path, index=False)

    # 3) python snippet
    py_path = log_dir / "feeder_bus_sets_stage2_8.py_snippet.txt"
    snippet = "FEEDER_BUS_SETS = " + json.dumps(feeder_bus_sets, ensure_ascii=False, indent=2) + "\n"
    py_path.write_text(snippet, encoding="utf-8")

    print(f"[OK] saved: {json_path}")
    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {py_path}")


def try_write_back_to_tie_map(project_root: Path, feeder_bus_sets: Dict[int, List[int]]):
    """
    把 topology/tie_map.py 里的 FEEDER_BUS_SETS = {...} 覆盖写回去
    （只替换那一行/那一段，方便追溯）
    """
    tie_map_path = project_root / "topology" / "tie_map.py"
    if not tie_map_path.exists():
        raise FileNotFoundError(f"cannot find {tie_map_path}")

    content = tie_map_path.read_text(encoding="utf-8")
    marker = "FEEDER_BUS_SETS"
    if marker not in content:
        raise RuntimeError("tie_map.py does not contain FEEDER_BUS_SETS marker")

    new_block = "FEEDER_BUS_SETS = " + json.dumps(feeder_bus_sets, ensure_ascii=False, indent=2)

    lines = content.splitlines()
    out_lines = []
    replaced = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("FEEDER_BUS_SETS"):
            # 替换这一行，且跳过可能的旧 dict 多行（简单做法：一直跳过到遇到空行或下一段注释/变量）
            out_lines.append(new_block)
            replaced = True
            i += 1
            # 跳过旧块（如果旧的是 {} 或者多行 dict）
            while i < len(lines):
                s = lines[i].strip()
                if s == "":
                    out_lines.append(lines[i])
                    i += 1
                    break
                # 如果下一行看起来像新的变量/注释段落，停止跳过
                if s.startswith("#") or ("=" in s and not s.startswith("{") and not s.startswith("}")):
                    break
                i += 1
            continue
        out_lines.append(line)
        i += 1

    if not replaced:
        raise RuntimeError("failed to replace FEEDER_BUS_SETS in tie_map.py")

    tie_map_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote back FEEDER_BUS_SETS into: {tie_map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-tie-map", action="store_true", help="write FEEDER_BUS_SETS back into topology/tie_map.py")
    args = parser.parse_args()

    # 约定：从 LLM_GP_Project 根目录运行
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"

    net = pn.mv_oberrhein()
    adj = build_electrical_adjacency(net)

    comps = connected_components_from_adj(adj)
    print(f"[INFO] components={len(comps)}, sizes={[len(c) for c in comps[:8]]}")

    # root bus 诊断
    rows = []
    bus_to_comp = {}
    for cid, comp in enumerate(comps):
        for b in comp:
            bus_to_comp[int(b)] = cid

    for feeder_id, rb in ROOT_BUSES.items():
        rows.append({"feeder_id": feeder_id, "root_bus": rb, "component_id": bus_to_comp.get(int(rb), -1)})
    print("\n=== ROOT BUS -> COMPONENT ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # 多源 BFS 分区
    owner = multi_source_bfs_partition(adj, ROOT_BUSES)
    feeder_bus_sets = invert_owner_to_sets(owner)

    print("\n=== FEEDER BUS SETS SUMMARY ===")
    for fid in sorted(feeder_bus_sets.keys()):
        print(f"feeder {fid}: n_bus={len(feeder_bus_sets[fid])}, sample={feeder_bus_sets[fid][:20]}")

    # 额外：检查 tie_map.TIE_SWITCHES 的端点是否跨 feeder
    try:
        from topology.tie_map import TIE_SWITCHES
        print("\n=== TIE_SWITCHES cross-feeder check ===")
        tie_rows = []
        for sw_id, (home, target) in TIE_SWITCHES.items():
            sw_id = int(sw_id)
            if sw_id not in net.switch.index:
                tie_rows.append({"switch_id": sw_id, "exists": False, "message": "switch not in net.switch"})
                continue
            if net.switch.at[sw_id, "et"] != "l":
                tie_rows.append({"switch_id": sw_id, "exists": True, "message": "not a line-switch"})
                continue
            line_id = int(net.switch.at[sw_id, "element"])
            fb = int(net.line.at[line_id, "from_bus"])
            tb = int(net.line.at[line_id, "to_bus"])
            f_fb = owner.get(fb, -1)
            f_tb = owner.get(tb, -1)
            cross = (f_fb != f_tb)
            tie_rows.append({
                "switch_id": sw_id,
                "line_id": line_id,
                "from_bus": fb,
                "to_bus": tb,
                "bus_feeder_from": f_fb,
                "bus_feeder_to": f_tb,
                "cross_feeder": cross,
                "declared_home": home,
                "declared_target": target,
                "sw_closed_now": bool(net.switch.at[sw_id, "closed"])
            })
        tie_df = pd.DataFrame(tie_rows)
        print(tie_df.to_string(index=False))

        tie_csv = log_dir / "debug_tie_switch_cross_feeder_stage2_8.csv"
        tie_df.to_csv(tie_csv, index=False)
        print(f"[OK] saved: {tie_csv}")
    except Exception as e:
        print(f"[WARN] cannot import topology.tie_map.TIE_SWITCHES for cross-check: {e}")

    # 保存留痕产物
    save_artifacts(log_dir, feeder_bus_sets, owner)

    # 可选：写回 tie_map.py
    if args.write_tie_map:
        try_write_back_to_tie_map(project_root, feeder_bus_sets)


if __name__ == "__main__":
    main()
