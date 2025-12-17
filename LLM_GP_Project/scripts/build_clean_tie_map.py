# scripts/build_clean_tie_map.py
import datetime as dt
import pandapower.networks as pn

def main():
    net = pn.mv_oberrhein(
        scenario="generation",
        cosphi_load=0.98,
        cosphi_pv=1.0,
        include_substations=False,
        separation_by_sub=False,
    )

    sw = net.switch
    line = net.line

    # 线路开关
    sw_l = sw[sw["et"] == "l"].copy()
    sw_l["line_id"] = sw_l["element"]

    # 按 line_id 分组，找“恰好两端开关”且“一开一合”的线路
    clean_ties = []
    for line_id, g in sw_l.groupby("line_id"):
        if len(g) != 2:
            continue
        closed = list(map(bool, g["closed"].values))
        # 一开一合：sum(closed) == 1
        if sum(closed) != 1:
            continue

        sw_ids = list(map(int, g.index.values))
        buses = list(map(int, g["bus"].values))
        fb = int(line.at[line_id, "from_bus"])
        tb = int(line.at[line_id, "to_bus"])

        clean_ties.append({
            "line": int(line_id),
            "switch_pair": sw_ids,     # 两端开关 id
            "switch_buses": buses,     # 两端开关所在 bus
            "line_ends": [fb, tb],     # 线路两端 bus
        })

    clean_ties = sorted(clean_ties, key=lambda d: d["line"])
    print(f"[CLEAN TIES] count={len(clean_ties)} (two switches and exactly one open)")

    # 输出成一个可直接复制的 python 文件（替换 topology/tie_map.py 用）
    out_path = "topology/tie_map.py"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n")
        f.write(f"# Auto-generated at {stamp}\n")
        f.write("# Only keeps line switch pairs that are 'one-open-one-closed' in baseline.\n\n")
        f.write("TIE_DEFS = [\n")
        for d in clean_ties:
            f.write(f"    {d},\n")
        f.write("]\n")

    print(f"[OK] wrote: {out_path}")
    print("Now re-run: python -m scripts.verify_tie_map")

if __name__ == "__main__":
    main()
