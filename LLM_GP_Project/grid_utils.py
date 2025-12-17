# grid_utils.py
import numpy as np
import pandapower as pp


def switch_stats(net):
    """统计开关情况（开关=能断/合的电气开断点）"""
    if net.switch is None or net.switch.empty:
        return {"n": 0, "open": 0, "closed": 0}

    n = len(net.switch)
    closed = int(net.switch["closed"].sum())
    open_ = n - closed
    return {"n": n, "open": open_, "closed": closed}


def print_switch_stats(net, title="Switch stats"):
    st = switch_stats(net)
    print(f"\n=== {title} ===")
    print(f"switch总数: {st['n']}, 常开(open): {st['open']}, 常闭(closed): {st['closed']}")


def _pick_two_far_buses(net, rng, n_candidates=20):
    """
    选两个“比较靠末端”的bus来做联络（近似方法：随机多选一些，偏向高索引/末端）。
    这里不强依赖拓扑算法，保证脚本可跑；你后面要更严格可改成按电气距离挑选。
    """
    buses = net.bus.index.values
    if len(buses) < 2:
        raise ValueError("bus数量不足，无法添加联络线。")

    # 候选：随机取一些，再从里面挑两个
    cand = rng.choice(buses, size=min(n_candidates, len(buses)), replace=False)
    b1, b2 = rng.choice(cand, size=2, replace=False)
    return int(b1), int(b2)


def ensure_tie_switch(net, n_ties=1, seed=42):
    """
    确保网络里至少存在 n_ties 个“常开联络开关”（tie switch）。
    做法：加一条“联络线”，在线的一端串一个 line switch，默认 closed=False（常开）。
    """
    rng = np.random.default_rng(seed)

    # 如果已经有常开开关，且数量够，就不动
    if net.switch is not None and (net.switch["closed"] == False).sum() >= n_ties:
        return []

    created = []
    for _ in range(n_ties):
        b1, b2 = _pick_two_far_buses(net, rng)

        # 避免重复连接
        existing = (
            ((net.line["from_bus"] == b1) & (net.line["to_bus"] == b2))
            | ((net.line["from_bus"] == b2) & (net.line["to_bus"] == b1))
        )
        if existing.any():
            continue

        # 选一个现成std_type，避免你手动填参数
        std_types = list(net.std_types.get("line", {}).keys())
        if not std_types:
            # 兜底：用参数建一条“虚拟联络线”
            line_idx = pp.create_line_from_parameters(
                net,
                from_bus=b1,
                to_bus=b2,
                length_km=0.2,
                r_ohm_per_km=0.4,
                x_ohm_per_km=0.25,
                c_nf_per_km=200,
                max_i_ka=0.3,
                name="TIE_LINE_VIRTUAL",
            )
        else:
            line_idx = pp.create_line(
                net,
                from_bus=b1,
                to_bus=b2,
                length_km=0.2,
                std_type=std_types[0],
                name="TIE_LINE",
            )

        # 在线路 b1 侧加一个开关，默认常开（closed=False）
        sw_idx = pp.create_switch(
            net,
            bus=b1,
            element=line_idx,
            et="l",
            closed=False,
            type="LBS",
            name="TIE_SWITCH_OPEN",
        )
        created.append((line_idx, sw_idx))

    return created


def scale_some_pv(net, factor=1.5, share=0.3, seed=42):
    """
    把一部分光伏容量调大，制造电压越上限风险（电压越限=电压超过允许范围）。
    """
    if net.sgen is None or net.sgen.empty:
        print("[WARN] net.sgen为空：没有分布式电源(如光伏)可调整。")
        return []

    rng = np.random.default_rng(seed)
    idx = net.sgen.index.values
    k = max(1, int(len(idx) * share))
    chosen = rng.choice(idx, size=k, replace=False)

    net.sgen.loc[chosen, "p_mw"] *= factor
    return chosen.tolist()
