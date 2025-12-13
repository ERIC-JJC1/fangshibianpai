
import pandapower.networks as pn
import pandapower.plotting as pp
import networkx as nx

net = pn.mv_oberrhein()  # 换成你的网
G = pp.create_nxgraph(net)  # 默认无向图

def traverse_feeder(G, start_bus, first_neighbor, exclude_buses):
    visited = set([start_bus, first_neighbor])
    stack = [first_neighbor]
    while stack:
        current = stack.pop()
        for nb in G.neighbors(current): 
            # 不能往回走start_bus，也不能进黑名单节点
            if nb not in visited and nb not in exclude_buses:
                visited.add(nb)
                stack.append(nb)
    return sorted(visited)

# 1. 第一条馈线
feeder1 = traverse_feeder(G, 319, 6, {147})
# 2. 第二条馈线
feeder2 = traverse_feeder(G, 319, 126, {190, 132, 54, 223})
# 3. 第三条馈线
feeder3 = traverse_feeder(G, 58, 86, {45, 195, 236, 80})
# 4. 第四条馈线
feeder4 = traverse_feeder(G, 80, 117, {39, 35, 31})

# ------ 变更分配（按你要求） ------
if 119 not in feeder4: feeder4.append(119)
if 318 not in feeder1: feeder1.append(318)
if 319 in feeder2: feeder2.remove(319)
feeder1 = sorted(feeder1)
feeder2 = sorted(feeder2)
feeder3 = sorted(feeder3)
feeder4 = sorted(feeder4)

# 4条馈线节点集合  
feeder1_set = set(feeder1)  
feeder2_set = set(feeder2)  
feeder3_set = set(feeder3)  
feeder4_set = set(feeder4)  

# 合并所有节点  
all_feeders = [feeder1_set, feeder2_set, feeder3_set, feeder4_set]  
all_nodes = feeder1_set | feeder2_set | feeder3_set | feeder4_set  

# 检查每对馈线之间是否有交集  
for i in range(4):  
    for j in range(i+1, 4):  
        overlap = all_feeders[i] & all_feeders[j]  
        if overlap:  
            print(f"馈线{i+1}和馈线{j+1}的重复节点（bus index）：{sorted(list(overlap))}")  

# 检查整体是否有重复节点
total_len = len(feeder1) + len(feeder2) + len(feeder3) + len(feeder4)  
unique_len = len(all_nodes)  
if unique_len < total_len:  
    print(f"总共有{total_len - unique_len}个节点在多个馈线中重复。")  
else:  
    print("四条馈线之间没有重复节点。") 

feeder_covered_nodes = all_nodes  
print(f"所有馈线覆盖（去重后）一共有 {len(feeder_covered_nodes)} 个节点(bus index)")  

network_nodes = set(G.nodes)         # 整个网络的bus index集合  
print(f"整个网络一共有 {len(network_nodes)} 个节点(bus index)")  

missing_nodes = network_nodes - feeder_covered_nodes  
print(f"未被任何馈线覆盖的节点有：{sorted(list(missing_nodes))}")  

for n in sorted(missing_nodes):  
    neighbors = list(G.neighbors(n))  
    print(f"未分配节点 {n} 的直接相邻节点是：{neighbors}")


########设置重要节点#######
important_buses = [167, 273, 244, 65, 148, 216, 227]  
net.bus["important"] = False        # 先全员设为 False  
net.bus.loc[important_buses, "important"] = True 
print(net.bus['important'])  
print(len(net.bus))