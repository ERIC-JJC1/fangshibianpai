# data_gen_utils.py
import numpy as np
import pandas as pd


def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def _daily_load_pattern_by_type(load_type: str):
    """
    返回 24小时 标准化曲线（0~1附近），不同类型有不同形状。
    """
    h = np.arange(24)

    # 居民：早晚峰明显
    if load_type in ("residential", "residential_LV"):
        base = (
            0.25
            + 0.25 * np.exp(-0.5 * ((h - 8) / 2.2) ** 2)
            + 0.35 * np.exp(-0.5 * ((h - 19) / 2.8) ** 2)
        )
        base += 0.05 * np.exp(-0.5 * ((h - 13) / 3.5) ** 2)  # 午间少量活动

    # 商业：白天高、夜里低
    elif load_type == "commercial":
        base = 0.18 + 0.65 * np.exp(-0.5 * ((h - 14) / 4.0) ** 2)

    # 工业：相对平稳，白天略高
    elif "industrial" in load_type:
        base = 0.55 + 0.15 * np.exp(-0.5 * ((h - 13) / 5.0) ** 2)

    else:
        base = 0.35 + 0.25 * np.exp(-0.5 * ((h - 18) / 4.0) ** 2)

    # 归一到大约 0.2~1.0
    base = base / base.max()
    base = 0.2 + 0.8 * base
    return base


def infer_load_type(name: str):
    name = str(name)
    if "CI" in name:
        return "commercial"
    if "MV" in name:
        return "industrial_MV"
    if "LV" in name:
        return "residential_LV"
    if "R" in name:
        return "residential"
    return "other"


def generate_load_profile(net, days=7, seed=42):
    """
    生成 load_profile.csv（行=load_idx，列=hour_0..hour_167）
    特点：
      - 类型曲线差异（居民/商业/工业）
      - “全局因子”相关性：同一天大家都受天气/温度影响（相关性=一起涨/一起跌）
      - 每个负荷再叠加少量本地扰动
    """
    rng = np.random.default_rng(seed)
    total_hours = days * 24

    # 全局因子：模拟“温度/宏观行为”驱动，同一天所有负荷一起偏高/偏低
    # 让它随天缓慢变化（更真实）
    global_day = rng.normal(1.0, 0.06, size=days)
    global_day = np.clip(global_day, 0.85, 1.15)

    rows = []
    for load_idx in net.load.index:
        row = {"load_idx": int(load_idx)}
        info = net.load.loc[load_idx]
        base_p = float(info["p_mw"])
        load_type = infer_load_type(info.get("name", ""))

        # 周末因子（周末=第6/7天）
        weekend_factor = 1.0
        if load_type == "commercial":
            weekend_factor = 0.6
        elif "industrial" in load_type:
            weekend_factor = 0.75
        else:
            weekend_factor = 0.85

        pattern = _daily_load_pattern_by_type(load_type)

        # 负荷自身长期偏差（有的人就是更“耗电”）
        local_scale = rng.normal(1.0, 0.05)
        local_scale = np.clip(local_scale, 0.85, 1.20)

        curve = []
        for d in range(days):
            is_weekend = d >= 5
            day_factor = weekend_factor if is_weekend else 1.0

            # 当天本地扰动（同一负荷当天整体偏高/偏低）
            local_day = rng.normal(1.0, 0.04)
            local_day = np.clip(local_day, 0.85, 1.15)

            for h in range(24):
                # 小时级噪声
                eps = rng.normal(0.0, 0.02)
                hour_factor = np.clip(1.0 + eps, 0.9, 1.1)

                p = base_p * local_scale * global_day[d] * local_day * day_factor * pattern[h] * hour_factor
                p = max(0.05 * base_p, p)
                curve.append(p)

        for t in range(total_hours):
            row[f"hour_{t}"] = curve[t]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("load_idx")
    return df


def _clear_sky_shape():
    """简单晴空曲线：6~18点为正弦，其他为0"""
    h = np.arange(24)
    shape = np.zeros(24)
    mask = (h >= 6) & (h <= 18)
    x = (h[mask] - 6) / 12.0  # 0..1
    shape[mask] = np.sin(np.pi * x)
    return shape


def generate_pv_profile(net, days=7, seed=42):
    """
    生成 pv_profile.csv（行=sgen_idx，列=hour_0..）
    特点（关键的可信度点）：
      - 全局天气 global_weather：空间强相关（同城差不多天）
      - 局部扰动 local_weather：每个点略有差异
      - 云遮挡 cloud_series：小时级随机，但对所有PV有共同影响（相关性）
    """
    if net.sgen is None or net.sgen.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    total_hours = days * 24
    clear = _clear_sky_shape()

    # 每天全局天气：0.5~1.0（晴/阴）
    global_weather_day = rng.uniform(0.55, 1.00, size=days)

    # 小时级“云遮挡”全局过程：用 AR(1) 生成连续波动（更像云团经过）
    cloud = []
    x = 0.0
    for _ in range(total_hours):
        x = 0.85 * x + rng.normal(0, 0.12)
        cloud.append(x)
    cloud = np.array(cloud)
    # 映射到 0.6~1.05 的乘子
    global_cloud_hour = _clip(1.0 - 0.25 * cloud, 0.60, 1.05)

    rows = []
    for sgen_idx in net.sgen.index:
        row = {"sgen_idx": int(sgen_idx)}
        rated = float(net.sgen.loc[sgen_idx, "p_mw"])

        # 每个点的局部天气偏差（叠加到全局上）
        local_bias = rng.normal(0.0, 0.05)

        curve = []
        for t in range(total_hours):
            d = t // 24
            h = t % 24

            # 局部天气=全局天气 + 小扰动
            local_weather = global_weather_day[d] + local_bias + rng.normal(0, 0.02)
            local_weather = float(np.clip(local_weather, 0.35, 1.10))

            # 真实PV：夜间为0，白天=额定*晴空*天气*云遮挡
            p = rated * clear[h] * local_weather * global_cloud_hour[t]

            # 小扰动（逆变器/测量/局部遮挡）
            p *= float(np.clip(1.0 + rng.normal(0, 0.015), 0.95, 1.05))

            p = max(0.0, min(rated, p))
            curve.append(p)

        for t in range(total_hours):
            row[f"hour_{t}"] = curve[t]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("sgen_idx")
    return df


def apply_profiles_to_net(net, load_df, pv_df, hour_idx: int):
    """
    把某个小时的profile写回net，用于 runpp 验证。
    """
    if not load_df.empty:
        for load_idx in net.load.index:
            net.load.loc[load_idx, "p_mw"] = float(load_df.loc[int(load_idx), f"hour_{hour_idx}"])

    if pv_df is not None and not pv_df.empty and net.sgen is not None and not net.sgen.empty:
        for sgen_idx in net.sgen.index:
            net.sgen.loc[sgen_idx, "p_mw"] = float(pv_df.loc[int(sgen_idx), f"hour_{hour_idx}"])
