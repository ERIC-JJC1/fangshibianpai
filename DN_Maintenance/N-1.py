import pandapower as pp  
import pandapower.networks as pn  


net = pn.mv_oberrhein()  
# 找所有主干支路，一般是两端电压较高、连接负荷多的line  
main_lines = net.line[(net.line['length_km'] > 0.3) & (net.line['in_service'] == True)].index.tolist()  

# 或者找“覆盖负荷最多”的前几条line（需统计line与其侧bus的负荷）  
# 假设有函数get_load_count_on_line  
def get_connected_load_count(net, line_idx):  
    from_bus = net.line.at[line_idx, 'from_bus']  
    to_bus = net.line.at[line_idx, 'to_bus']  
    load_count = ((net.load['bus'] == from_bus) | (net.load['bus'] == to_bus)).sum()  
    return load_count  

lines_load_counts = {idx: get_connected_load_count(net, idx) for idx in net.line.index}  
lines_sorted = sorted(lines_load_counts.items(), key=lambda x: -x[1])  
# 假设只挑前5条  
n1_candidates = [idx for idx, cnt in lines_sorted[:5]]

print(net.line.loc[n1_candidates, ["from_bus", "to_bus", "length_km"]])  