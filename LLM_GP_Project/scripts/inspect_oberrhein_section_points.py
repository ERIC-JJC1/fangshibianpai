# scripts/inspect_oberrhein_section_points.py
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

def main():
    print("[INFO] pandapower version:", pp.__version__)

    # 跟你 verify 脚本一致的建网方式（generation 场景）
    net = pn.mv_oberrhein(
        scenario="generation",
        cosphi_load=0.98,
        cosphi_pv=1.0,
        include_substations=False,
        separation_by_sub=False,
    )

    sw = net.switch.copy()
    line = net.line.copy()

    # 只看线路开关（et == 'l'）
    sw_l = sw[sw["et"] == "l"].copy()
    sw_l["line_id"] = sw_l["element"]

    # 把线路两端的 bus 信息拼进去
    sw_l["from_bus"] = sw_l["line_id"].map(line["from_bus"])
    sw_l["to_bus"] = sw_l["line_id"].map(line["to_bus"])

    # 找出“至少有一个端开关是 open”的线路 —— 这类就是候选 sectioning point / 常开点
    grp = sw_l.groupby("line_id")["closed"].apply(list).reset_index(name="closed_list")
    open_lines = grp[grp["closed_list"].apply(lambda xs: any([not bool(x) for x in xs]))]["line_id"].tolist()

    print(f"\n[SECTION POINT CANDIDATES] lines with ANY open switch: {len(open_lines)}")
    # 把这些线路的开关都打印出来（按 line_id 排序）
    cand = sw_l[sw_l["line_id"].isin(open_lines)].sort_values(["line_id", "bus"])

    # 更友好一点的显示
    show_cols = ["line_id", "bus", "from_bus", "to_bus", "closed"]
    print(cand[show_cols].to_string(index=True))

    # 统计每条线路有几个开关、是否“两端一开一合”
    stat = sw_l[sw_l["line_id"].isin(open_lines)].groupby("line_id").agg(
        n_switch=("bus", "count"),
        n_open=("closed", lambda s: int((~s.astype(bool)).sum())),
        n_closed=("closed", lambda s: int((s.astype(bool)).sum())),
        buses=("bus", lambda s: list(map(int, s.values))),
    ).reset_index().sort_values("line_id")

    print("\n[SUMMARY]")
    print(stat.to_string(index=False))

    # 保存一份 csv，留痕可追溯
    out = "logs/oberrhein_section_points.csv"
    stat.to_csv(out, index=False)
    print(f"\n[OK] saved: {out}")

if __name__ == "__main__":
    main()
