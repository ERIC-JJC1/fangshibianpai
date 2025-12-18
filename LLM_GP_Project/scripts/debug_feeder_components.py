import os
import pandas as pd
import pandapower.networks as pn
import pandapower.topology as top

from topology.tie_map import TIE_SWITCHES

FEEDER_ROOT_BUSES = {
    1: 319,
    2: 126,
    3: 58,
    4: 80,
}

def force_section_points_open(net):
    for sw_id in TIE_SWITCHES.keys():
        net.switch.at[sw_id, "closed"] = False

def component_id_of_bus(components, bus):
    for i, comp in enumerate(components):
        if bus in comp:
            return i
    return None

def main():
    os.makedirs("logs", exist_ok=True)

    net = pn.mv_oberrhein()
    force_section_points_open(net)

    G = top.create_nxgraph(net, respect_switches=True)
    comps = list(top.connected_components(G))
    comps = sorted(comps, key=len, reverse=True)

    print(f"[INFO] components={len(comps)}, sizes={[len(c) for c in comps]}")

    # 1) root buses在哪个component
    rows = []
    for fid, rb in FEEDER_ROOT_BUSES.items():
        cid = component_id_of_bus(comps, rb)
        rows.append({"feeder_id": fid, "root_bus": rb, "component_id": cid})
    df_roots = pd.DataFrame(rows)
    print("\n=== ROOT BUS -> COMPONENT ===")
    print(df_roots.to_string(index=False))

    # 2) 把 TIE_SWITCHES 每个开关两端bus的component打印出来（看它是否真的跨分量）
    tie_rows = []
    for sw_id in TIE_SWITCHES.keys():
        sw = net.switch.loc[sw_id]
        if sw["et"] != "l":
            continue
        line_id = int(sw["element"])
        fb = int(net.line.at[line_id, "from_bus"])
        tb = int(net.line.at[line_id, "to_bus"])
        c_fb = component_id_of_bus(comps, fb)
        c_tb = component_id_of_bus(comps, tb)
        tie_rows.append({
            "switch_id": sw_id,
            "line_id": line_id,
            "from_bus": fb,
            "to_bus": tb,
            "sw_closed": bool(sw["closed"]),
            "comp_from": c_fb,
            "comp_to": c_tb,
            "cross_component": (c_fb is not None and c_tb is not None and c_fb != c_tb)
        })
    df_ties = pd.DataFrame(tie_rows).sort_values(["cross_component","switch_id"], ascending=[False, True])
    print("\n=== TIE_SWITCHES cross-component check ===")
    print(df_ties.to_string(index=False))

    out_csv = "logs/debug_components_ties_stage2_8.csv"
    df_ties.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}")

if __name__ == "__main__":
    main()
