# topology/tie_registry.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


def _extract_pairs(tie_map: Dict) -> List[Tuple[int, int, int]]:
    """
    返回 [(tie_id, sw_a, sw_b), ...]
    tie_id 用 line_id（如果没有就用顺序号）
    """
    seen = set()
    out = []
    for _, links in tie_map.items():
        for link in links:
            sw1, sw2 = link["switch_pair"]
            tie_id = int(link.get("line", -1))
            key = tuple(sorted((int(sw1), int(sw2))) + [tie_id])
            if key in seen:
                continue
            seen.add(key)
            out.append((tie_id, int(sw1), int(sw2)))
    # tie_id=-1 的排后面
    return sorted(out, key=lambda x: (x[0] == -1, x[0], x[1], x[2]))


def annotate_tie_switches(net, tie_map: Dict) -> None:
    """
    给 net.switch 增加:
      - tie_id: 每对联络开关共享一个编号（默认用 line_id）
      - tie_peer_switch: 对端开关ID
    """
    if "tie_id" not in net.switch.columns:
        net.switch["tie_id"] = pd.NA
    if "tie_peer_switch" not in net.switch.columns:
        net.switch["tie_peer_switch"] = pd.NA

    pairs = _extract_pairs(tie_map)
    for tie_id, sw_a, sw_b in pairs:
        if sw_a in net.switch.index:
            net.switch.at[sw_a, "tie_id"] = tie_id
            net.switch.at[sw_a, "tie_peer_switch"] = sw_b
        if sw_b in net.switch.index:
            net.switch.at[sw_b, "tie_id"] = tie_id
            net.switch.at[sw_b, "tie_peer_switch"] = sw_a


def validate_inter_feeder_switch_map(net, tie_map: Dict) -> pd.DataFrame:
    """
    返回 issues 明细 DataFrame：
    columns: level, tie_id, switch_id, peer_switch_id, line_id, bus, rule, message
    """
    issues: List[dict] = []
    pairs = _extract_pairs(tie_map)

    def add(level: str, tie_id: int, sw: int, peer: int, line_id: Optional[int], bus: Optional[int],
            rule: str, message: str):
        issues.append({
            "level": level,
            "tie_id": tie_id,
            "switch_id": sw,
            "peer_switch_id": peer,
            "line_id": line_id,
            "bus": bus,
            "rule": rule,
            "message": message,
        })

    for tie_id, sw_a, sw_b in pairs:
        # 规则 1：开关ID存在
        if sw_a not in net.switch.index:
            add("ERROR", tie_id, sw_a, sw_b, tie_id if tie_id != -1 else None, None,
                "switch_exists", "switch_id not found in net.switch")
            continue
        if sw_b not in net.switch.index:
            add("ERROR", tie_id, sw_b, sw_a, tie_id if tie_id != -1 else None, None,
                "switch_exists", "peer_switch_id not found in net.switch")
            continue

        row_a = net.switch.loc[sw_a]
        row_b = net.switch.loc[sw_b]
        bus_a = int(row_a["bus"]) if pd.notna(row_a["bus"]) else None
        bus_b = int(row_b["bus"]) if pd.notna(row_b["bus"]) else None

        # 规则 2：必须是线路开关（et == 'l'）
        if str(row_a.get("et", "")) != "l":
            add("WARN", tie_id, sw_a, sw_b, int(row_a.get("element")) if pd.notna(row_a.get("element")) else None, bus_a,
                "line_switch", f"expected et='l', got et='{row_a.get('et')}'")
        if str(row_b.get("et", "")) != "l":
            add("WARN", tie_id, sw_b, sw_a, int(row_b.get("element")) if pd.notna(row_b.get("element")) else None, bus_b,
                "line_switch", f"expected et='l', got et='{row_b.get('et')}'")

        # 规则 3：两端应指向同一条线路 element（更符合“一条联络线两端开关”的建模）
        elem_a = int(row_a["element"]) if pd.notna(row_a.get("element")) else None
        elem_b = int(row_b["element"]) if pd.notna(row_b.get("element")) else None
        if elem_a is not None and elem_b is not None and elem_a != elem_b:
            add("WARN", tie_id, sw_a, sw_b, elem_a, bus_a, "same_line_element",
                f"peer switches refer to different line elements: {elem_a} vs {elem_b}")
            add("WARN", tie_id, sw_b, sw_a, elem_b, bus_b, "same_line_element",
                f"peer switches refer to different line elements: {elem_b} vs {elem_a}")

        # 规则 4：peer 关系自洽
        peer_a = row_a.get("tie_peer_switch", pd.NA)
        peer_b = row_b.get("tie_peer_switch", pd.NA)
        if pd.notna(peer_a) and int(peer_a) != sw_b:
            add("WARN", tie_id, sw_a, sw_b, elem_a, bus_a, "peer_consistency",
                f"tie_peer_switch mismatch: annotated={int(peer_a)}, expected={sw_b}")
        if pd.notna(peer_b) and int(peer_b) != sw_a:
            add("WARN", tie_id, sw_b, sw_a, elem_b, bus_b, "peer_consistency",
                f"tie_peer_switch mismatch: annotated={int(peer_b)}, expected={sw_a}")

        # 规则 5：运行语义（不是硬错误，但要标出来）
        closed_a = bool(row_a.get("closed", True))
        closed_b = bool(row_b.get("closed", True))
        if closed_a and closed_b:
            # 两端都合：更像“正在并列运行”，不再是典型常开联络点
            add("WARN", tie_id, sw_a, sw_b, elem_a, bus_a, "operational_semantics",
                "both ends are closed -> tie is electrically connected (may create loop).")
            add("WARN", tie_id, sw_b, sw_a, elem_b, bus_b, "operational_semantics",
                "both ends are closed -> tie is electrically connected (may create loop).")
        elif (not closed_a) and (not closed_b):
            # 两端都断：也许合理（两端隔离），但要提示
            add("WARN", tie_id, sw_a, sw_b, elem_a, bus_a, "operational_semantics",
                "both ends are open -> tie is fully isolated (ok if intended).")
            add("WARN", tie_id, sw_b, sw_a, elem_b, bus_b, "operational_semantics",
                "both ends are open -> tie is fully isolated (ok if intended).")

    return pd.DataFrame(issues)
