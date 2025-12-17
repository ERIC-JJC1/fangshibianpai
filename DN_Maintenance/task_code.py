#此版本代码是2025/5/15改动的代码

#============================================================================= 
#输入必要的库
import random  
import pandas as pd  
import numpy as np  
from datetime import datetime, timedelta  
import copy  
import pandapower.networks as pn  
import pandapower as pp
import pandapower.topology as top
import pandapower.plotting as plot  # 如果你真需要画图 
import networkx as nx  
import copy  
#============================================================================= 
net = pn.mv_oberrhein()  
G = top.create_nxgraph(net, respect_switches=True)

def traverse_feeder(G, start_bus, first_neighbor, exclude_buses):  
    visited = set([start_bus, first_neighbor])  
    stack = [first_neighbor]  
    while stack:  
        current = stack.pop()  
        for nb in G.neighbors(current):  
            if nb not in visited and nb not in exclude_buses:  
                visited.add(nb)  
                stack.append(nb)  
    return sorted(visited)  

# 馈线参数根据你的现场网络实际调整  
feeder1 = traverse_feeder(G, 319, 6, {147})  
feeder2 = traverse_feeder(G, 126, 29, {190, 132, 54, 223})  
feeder3 = traverse_feeder(G, 58, 86, {45, 195, 236, 80})  
feeder4 = traverse_feeder(G, 80, 117, {39, 35, 31})  

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


all_feeders = [feeder1, feeder2, feeder3, feeder4]  
feeder_map = dict()  
for idx, feeder in enumerate(all_feeders, 1):  
    for bus in feeder:  
        feeder_map[bus] = idx  # bus index: feeder编号  


# 假设 feeder_map 已经构建好: {bus_id: feeder_id}  
net_switch = net.switch  
net_line = net.line  

inter_feeder_switches = []  
for sid, sw in net_switch.iterrows():  
    # 只考虑线路开关（联络多为line之上的开关）  
    if sw['et'] != 'l':  
        continue  
    line_id = sw['element']  
    if line_id not in net_line.index:  
        continue  
    from_bus = net_line.at[line_id, 'from_bus']  
    to_bus = net_line.at[line_id, 'to_bus']  
    feeder_from = feeder_map.get(from_bus)  
    feeder_to = feeder_map.get(to_bus)  
    # 两侧馈线均有编号且不同，才视为联络  
    if feeder_from is not None and feeder_to is not None and feeder_from != feeder_to:  
        inter_feeder_switches.append({  
            'switch_id': sid,  
            'line_id': line_id,  
            'from_bus': from_bus,  
            'to_bus': to_bus,  
            'from_feeder': feeder_from,  
            'to_feeder': feeder_to,  
            'closed': sw['closed'] if 'closed' in sw else None,  
        })  

# 打印结果  
print(f"\n找到{len(inter_feeder_switches)}个馈线间联络开关：")  
for s in inter_feeder_switches:  
    print(f"- 开关ID {s['switch_id']}: 线路 {s['line_id']} ({s['from_bus']}[馈线{s['from_feeder']}] ↔ {s['to_bus']}[馈线{s['to_feeder']}]), 状态: {'合' if s['closed'] else '分'}")



def generate_realistic_maintenance_plan(  
    net,  
    n_nodes,         # 节点级任务数量  
    n_single_lines,  # 单独线路任务数量  
    n_single_switches, # 单独开关任务数量   
    n_single_trafos,  # 单独变压器任务数量  
    start_date="2025-05-01",  
    days_span=7,      # 计划跨度天数  
    seed=None,  
    validate_with_powerflow=False,  # 是否使用潮流计算验证  
    resolve_conflicts=False        # 是否自动解决冲突  
):  
    """  
    生成真实可执行的检修计划，包括:  
    1. 节点级检修任务：每个任务包含节点相关的所有设备  
    2. 单设备检修任务：针对单个线路、开关或变压器的检修  
    3. 标记负荷转供需求：对于导致负荷失电的检修，标记需要转供  
    4. 任务冲突检测和解决：识别并解决任务间的冲突  
    5. 电气潮流验证：验证检修方案的可行性  
    """  
    random.seed(seed)  
    tasks = []  
    task_id = 1  
    
    # 添加辅助函数：负荷重要性判定  
    def determine_load_importance(load_id, net):  
        """确定负荷重要性级别"""  
        # 实际项目中应从设备台账或负荷表中获取  
        # 简化处理，可以根据负荷类型或容量判断
          
        if hasattr(net.load, 'type') and 'type' in net.load.columns:  
            load_type = net.load.at[load_id, 'type']  
            if load_type in ['hospital', 'data_center', 'industrial_key']:  
                return "一级"  
            elif load_type in ['commercial', 'industrial']:  
                return "二级"  
        
        # 根据容量判断  
        if 'p_mw' in net.load.columns:  
            p = net.load.at[load_id, 'p_mw']  
            if p > 1.0:  # 例如大于1MW视为重要负荷  
                return "一级"  
            elif p > 0.5:  # 大于500kW视为次重要负荷  
                return "二级"  
        
        return "三级"  # 默认为三级负荷  
    
    # 改进的下游节点识别算法  
    def identify_downstream_nodes_feeder(net, from_node, feeder_nodes, disconnected_elements=None):  
        """只在指定feeder_nodes内做广度优先下游遍历，与联络状态无关"""  
        if disconnected_elements is None:  
            disconnected_elements = []  
        # 剩下基本完全照你原函数，只加一行：next_node in feeder_nodes  
        disconnected_dict = {}  
        for elem_type, elem_id in disconnected_elements:  
            if elem_type not in disconnected_dict:  
                disconnected_dict[elem_type] = set()  
            disconnected_dict[elem_type].add(elem_id)  
        visited = set([from_node])  
        queue = [from_node]  
        while queue:  
            current_node = queue.pop(0)  
            # 检查所有连接线路  
            if hasattr(net, 'line'):  
                for idx, line in net.line.iterrows():  
                    if 'line' in disconnected_dict and idx in disconnected_dict['line']:  
                        continue  
                    if not line.get('in_service', True):  
                        continue  
                    next_node = None  
                    if line['from_bus'] == current_node:  
                        next_node = line['to_bus']  
                    elif line['to_bus'] == current_node:  
                        next_node = line['from_bus']  
                    if next_node is not None and next_node in feeder_nodes:  
                        switches_closed = True  
                        if hasattr(net, 'switch'):  
                            for s_idx, switch in net.switch.iterrows():  
                                if switch['et'] == 'l' and switch['element'] == idx:  
                                    if 'switch' in disconnected_dict and s_idx in disconnected_dict['switch']:  
                                        switches_closed = False  
                                        break  
                                    if not switch.get('closed', True):  
                                        switches_closed = False  
                                        break  
                        if switches_closed and next_node not in visited:  
                            visited.add(next_node)  
                            queue.append(next_node)  
            if hasattr(net, 'trafo'):  
                for idx, trafo in net.trafo.iterrows():  
                    if 'trafo' in disconnected_dict and idx in disconnected_dict['trafo']:  
                        continue  
                    if not trafo.get('in_service', True):  
                        continue  
                    next_node = None  
                    if trafo['hv_bus'] == current_node:  
                        next_node = trafo['lv_bus']  
                    elif trafo['lv_bus'] == current_node:  
                        next_node = trafo['hv_bus']  
                    if next_node is not None and next_node in feeder_nodes:  
                        switches_closed = True  
                        if hasattr(net, 'switch'):  
                            for s_idx, switch in net.switch.iterrows():  
                                if switch['et'] == 't' and switch['element'] == idx:  
                                    if 'switch' in disconnected_dict and s_idx in disconnected_dict['switch']:  
                                        switches_closed = False  
                                        break  
                                    if not switch.get('closed', True):  
                                        switches_closed = False  
                                        break  
                        if switches_closed and next_node not in visited:  
                            visited.add(next_node)  
                            queue.append(next_node)  
        visited.remove(from_node)  
        return list(visited)
    
    # 添加辅助函数：检查转供路径可行性  
    def has_transfer_path_advanced(net, affected_buses, max_depth=3):  
        """  
        使用广度优先搜索检查是否存在可行的转供路径  
        
        参数:  
        - net: pandapower网络模型  
        - affected_buses: 受影响的母线ID列表  
        - max_depth: 最大搜索深度，防止在大网络中搜索过深  
        
        返回:  
        - 布尔值，表示是否存在可行的转供路径  
        - 如果存在，还返回可能的转供路径列表  
        """  
        # 标记所有受影响的母线  
        affected_set = set(affected_buses)  
        
        # 为每个受影响的母线寻找可能的转供路径  
        for start_bus in affected_buses:  
            # 使用队列进行广度优先搜索  
            queue = [(start_bus, [], 0)]  # (当前母线, 路径, 深度)  
            visited = set([start_bus])  
            
            while queue:  
                current_bus, path, depth = queue.pop(0)  
                
                # 如果达到最大深度，停止搜索  
                if depth >= max_depth:  
                    continue  
                
                # 检查所有连接到当前母线的线路  
                if hasattr(net, 'line'):  
                    for idx, line in net.line.iterrows():  
                        # 检查线路是否在运行状态  
                        if not line.get('in_service', True):  
                            continue  
                        
                        next_bus = None  
                        if line['from_bus'] == current_bus:  
                            next_bus = line['to_bus']  
                        elif line['to_bus'] == current_bus:  
                            next_bus = line['from_bus']  
                        
                        # 如果找到连接的母线，且不在受影响集合中  
                        if next_bus is not None and next_bus not in affected_set:  
                            # 检查线路上的开关状态  
                            switches_ok = True  
                            if hasattr(net, 'switch'):  
                                line_switches = net.switch[(net.switch['et'] == 'l') &   
                                                        (net.switch['element'] == idx)]  
                                if len(line_switches) > 0 and not all(line_switches['closed']):  
                                    switches_ok = False  
                            
                            if switches_ok:  
                                # 找到一条有效的转供路径  
                                new_path = path + [(current_bus, next_bus, idx)]  
                                return True, new_path  
                        
                        # 如果下一个母线已经访问过，或者在受影响集合中，跳过  
                        if next_bus is None or next_bus in visited or next_bus in affected_set:  
                            continue  
                        
                        # 检查线路上的开关状态  
                        switches_ok = True  
                        if hasattr(net, 'switch'):  
                            line_switches = net.switch[(net.switch['et'] == 'l') &   
                                                    (net.switch['element'] == idx)]  
                            if len(line_switches) > 0 and not all(line_switches['closed']):  
                                switches_ok = False  
                        
                        if switches_ok:  
                            # 继续探索这条路径  
                            new_path = path + [(current_bus, next_bus, idx)]  
                            queue.append((next_bus, new_path, depth + 1))  
                            visited.add(next_bus)  
        
        # 如果没有找到任何转供路径  
        return False, []  
    
    # 改进后的转供判定函数  
    def determine_transfer_need(net, affected_buses, affected_load_ids):  
        """更准确地判断是否需要负荷转供"""  
        if not affected_load_ids:  
            return {  
                "needs_transfer": False,  
                "transfer_reason": "无受影响负荷"  
            }  
        
        # 收集负荷详细信息  
        load_details = []  
        total_load = 0.0  
        
        for load_id in affected_load_ids:  
            if load_id in net.load.index:  
                load_p = net.load.at[load_id, 'p_mw'] if 'p_mw' in net.load.columns else 0.1  
                importance = determine_load_importance(load_id, net)  
                load_details.append({  
                    "id": load_id,  
                    "p_mw": load_p,  
                    "importance": importance  
                })  
                total_load += load_p  
        
        # 判断转供需求  
        needs_transfer = False  
        transfer_reason = ""  
        transfer_path = None  
        
        # 存在一级负荷 - 必须转供  
        critical_loads = [l for l in load_details if l["importance"] == "一级"]  
        if critical_loads:  
            has_path, path = has_transfer_path_advanced(net, affected_buses)  
            if has_path:  
                needs_transfer = True  
                transfer_reason = "包含一级重要负荷"  
                transfer_path = path  
            else:  
                needs_transfer = False  
                transfer_reason = "包含一级重要负荷但无可行转供路径"  
        
        # 存在总负荷超过阈值 - 需要转供  
        elif total_load > 0.5:  # 例如超过500kW  
            has_path, path = has_transfer_path_advanced(net, affected_buses)  
            if has_path:  
                needs_transfer = True  
                transfer_reason = f"总负荷({total_load:.2f}MW)超过阈值"  
                transfer_path = path  
            else:  
                needs_transfer = False  
                transfer_reason = f"总负荷({total_load:.2f}MW)超过阈值但无可行转供路径"  
        
        # 存在二级负荷且有可行转供路径 - 优先转供  
        elif any(l["importance"] == "二级" for l in load_details):  
            has_path, path = has_transfer_path_advanced(net, affected_buses)  
            if has_path:  
                needs_transfer = True  
                transfer_reason = "包含二级负荷且有可行转供路径"  
                transfer_path = path  
        
        # 无关键负荷但总负荷不为零且存在转供路径 - 考虑转供  
        elif total_load > 0:  
            has_path, path = has_transfer_path_advanced(net, affected_buses)  
            if has_path:  
                needs_transfer = True  
                transfer_reason = "存在普通负荷且有可行转供路径"  
                transfer_path = path  
        
        return {  
            "needs_transfer": needs_transfer,  
            "transfer_reason": transfer_reason,  
            "total_load_mw": total_load,  
            "load_details": load_details,  
            "transfer_path": transfer_path if needs_transfer else None  
        }  
    
    # 电气潮流验证检修方案  
    def validate_maintenance_plan_with_powerflow(net, tasks):  
        """  
        通过潮流计算验证检修方案的可行性  
        
        返回：  
        - 验证结果列表，包含每个任务的验证状态  
        - 包含违反约束的详细信息  
        """  
       
 
        results = []  
        
        # 深拷贝原始网络用于模拟  
        original_net = copy.deepcopy(net)  
        
        for task in tasks:  
            # 重置网络到原始状态  
            simulation_net = copy.deepcopy(original_net)  
            
            # 根据任务修改网络拓扑  
            if task["task_type"] == "node":  
                # 节点级任务 - 断开相关设备  
                for line_id in task["equipment"]["lines"]:  
                    if line_id in simulation_net.line.index:  
                        simulation_net.line.at[line_id, "in_service"] = False  
                
                for switch_id in task["equipment"]["switches"]:  
                    if switch_id in simulation_net.switch.index:  
                        simulation_net.switch.at[switch_id, "closed"] = False  
                
                for trafo_id in task["equipment"]["trafos"]:  
                    if trafo_id in simulation_net.trafo.index:  
                        simulation_net.trafo.at[trafo_id, "in_service"] = False  
            
            else:  # 单设备任务  
                element_type = task["element_type"]  
                element_id = task["element_id"]  
                
                if element_type == "line" and element_id in simulation_net.line.index:  
                    simulation_net.line.at[element_id, "in_service"] = False  
                
                elif element_type == "switch" and element_id in simulation_net.switch.index:  
                    simulation_net.switch.at[element_id, "closed"] = False  
                
                elif element_type == "trafo" and element_id in simulation_net.trafo.index:  
                    simulation_net.trafo.at[element_id, "in_service"] = False  
            
            # 执行潮流计算  
            try:  
                pp.runpp(simulation_net, numba=False)  
                
                # 检查约束违反  
                violations = {  
                    "voltage_violations": [],  
                    "loading_violations": [],  
                    "convergence": True  
                }  
                
                # 检查电压越限  
                v_max = simulation_net.bus["max_vm_pu"].values if "max_vm_pu" in simulation_net.bus.columns else np.ones(len(simulation_net.bus)) * 1.1  
                v_min = simulation_net.bus["min_vm_pu"].values if "min_vm_pu" in simulation_net.bus.columns else np.ones(len(simulation_net.bus)) * 0.9  
                
                for i, bus in simulation_net.bus.iterrows():  
                    vm = simulation_net.res_bus.at[i, "vm_pu"]  
                    if vm > v_max[i] or vm < v_min[i]:  
                        violations["voltage_violations"].append({  
                            "bus_id": i,  
                            "voltage": vm,  
                            "limit": f"{v_min[i] if vm < v_min[i] else v_max[i]}"  
                        })  
                
                # 检查线路过载  
                for i, line in simulation_net.line.iterrows():  
                    if line["in_service"]:  
                        loading = simulation_net.res_line.at[i, "loading_percent"]  
                        if loading > 100:  
                            violations["loading_violations"].append({  
                                "element_type": "line",  
                                "element_id": i,  
                                "loading": loading,  
                                "limit": 100  
                            })  
                
                # 检查变压器过载  
                for i, trafo in simulation_net.trafo.iterrows():  
                    if trafo["in_service"]:  
                        loading = simulation_net.res_trafo.at[i, "loading_percent"]  
                        if loading > 100:  
                            violations["loading_violations"].append({  
                                "element_type": "trafo",  
                                "element_id": i,  
                                "loading": loading,  
                                "limit": 100  
                            })  
                
                # 判断验证结果  
                is_valid = (len(violations["voltage_violations"]) == 0 and   
                            len(violations["loading_violations"]) == 0)  
                
                results.append({  
                    "task_id": task["task_id"],  
                    "is_valid": is_valid,  
                    "violations": violations  
                })  
                
            except Exception as e:  
                # 潮流不收敛或其他错误  
                results.append({  
                    "task_id": task["task_id"],  
                    "is_valid": False,  
                    "violations": {  
                        "voltage_violations": [],  
                        "loading_violations": [],  
                        "convergence": False,  
                        "error_message": str(e)  
                    }  
                })  
        
        return results  
    
    # 任务冲突检测  
    def detect_task_conflicts(tasks, net):  
        """检测任务间的潜在冲突和相互影响"""  
        conflicts = []  
        
        # 创建按日期时间分组的任务字典  
        tasks_by_time = {}  
        for task in tasks:  
            start_time = datetime.strptime(task["start_time"], "%Y-%m-%d %H:%M")  
            end_time = start_time + timedelta(hours=task["duration_h"])  
            
            # 将任务添加到其时间范围内的每一个小时  
            current = start_time  
            while current < end_time:  
                time_key = current.strftime("%Y-%m-%d %H")  
                if time_key not in tasks_by_time:  
                    tasks_by_time[time_key] = []  
                tasks_by_time[time_key].append(task["task_id"])  
                current += timedelta(hours=1)  
        
        # 分析每个时间点的任务组合是否存在冲突  
        for time_key, task_ids in tasks_by_time.items():  
            if len(task_ids) > 1:  # 同一时间有多个任务  
                # 获取该时间的任务列表  
                concurrent_tasks = [t for t in tasks if t["task_id"] in task_ids]  
                
                # 检查是否涉及同一区域或设备  
                affected_buses = set()  
                affected_equipment = {  
                    "lines": set(),  
                    "switches": set(),  
                    "trafos": set()  
                }  
                
                for t in concurrent_tasks:  
                    if t["task_type"] == "node":  
                        affected_buses.add(t["node_id"])  
                        for eq_type, eq_ids in t["equipment"].items():  
                            affected_equipment[eq_type].update(eq_ids)  
                    else:  # 单设备任务  
                        affected_buses.update(t.get("affected_buses", []))  
                        if t["element_type"] in affected_equipment:  
                            affected_equipment[t["element_type"]].add(t["element_id"])  
                
                # 计算影响总负荷  
                total_affected_load = sum(t.get("total_load_mw", 0) for t in concurrent_tasks)  
                
                # 检查是否有设备重叠  
                has_overlap = False  
                for eq_type, eq_ids in affected_equipment.items():  
                    if len(eq_ids) != sum(len(t["equipment"][eq_type]) if t["task_type"] == "node" else   
                                        (1 if t["element_type"] == eq_type else 0)   
                                        for t in concurrent_tasks):  
                        has_overlap = True  
                        break  
                
                # 检查是否有母线重叠  
                bus_overlap = len(affected_buses) < sum(1 if t["task_type"] == "node" else len(t.get("affected_buses", []))   
                                                    for t in concurrent_tasks)  
                
                # 如果存在重叠或总负荷太大，记录冲突  
                if has_overlap or bus_overlap or total_affected_load > 3.0:  # 阈值可调整  
                    conflicts.append({  
                        "time": time_key,  
                        "task_ids": task_ids,  
                        "has_equipment_overlap": has_overlap,  
                        "has_bus_overlap": bus_overlap,  
                        "total_affected_load": total_affected_load,  
                        "severity": "高" if total_affected_load > 5.0 else "中" if total_affected_load > 2.0 else "低"  
                    })  
        
        return conflicts  

    # 任务冲突解决    
    def resolve_task_conflicts(tasks, conflicts, net):  
        """尝试解决任务间的冲突"""  
        resolved_tasks = copy.deepcopy(tasks)  
        
        # 按冲突严重程度排序（先解决严重冲突）  
        sorted_conflicts = sorted(conflicts, key=lambda c: {"高": 3, "中": 2, "低": 1}[c["severity"]], reverse=True)  
        
        for conflict in sorted_conflicts:  
            # 获取冲突任务  
            conflict_tasks = [t for t in resolved_tasks if t["task_id"] in conflict["task_ids"]]  
            
            # 基于任务优先级排序  
            conflict_tasks.sort(key=lambda t: t["priority"])  
            
            # 尝试调整低优先级任务的时间  
            for i in range(1, len(conflict_tasks)):  
                task = conflict_tasks[i]  
                
                # 尝试向后推迟任务（最多推迟7天）  
                for delay_hours in [24, 48, 72, 96, 120, 144, 168]:  
                    # 计算新的开始时间  
                    start_time = datetime.strptime(task["start_time"], "%Y-%m-%d %H:%M")  
                    new_start_time = start_time + timedelta(hours=delay_hours)  
                    new_start_str = new_start_time.strftime("%Y-%m-%d %H:%M")  
                    
                    # 更新任务时间  
                    for t in resolved_tasks:  
                        if t["task_id"] == task["task_id"]:  
                            t["start_time"] = new_start_str  
                            t["desc"] += f"（已调整：解决任务冲突）"  
                            break  
                    
                    # 检查调整后是否仍有冲突  
                    new_conflicts = detect_task_conflicts([t for t in resolved_tasks   
                                                        if t["task_id"] in conflict["task_ids"]], net)  
                    if not any(c["time"] == conflict["time"] for c in new_conflicts):  
                        # 冲突已解决，退出循环  
                        break  
        
        return resolved_tasks  
    
    # 日期生成器  
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")  
    def get_random_datetime():  
        random_day = random.randint(0, days_span-1)  
        random_hour = random.randint(7, 17)  
        dt = start_dt + timedelta(days=random_day)  
        return f"{dt.strftime('%Y-%m-%d')} {random_hour:02d}:00"  
    
    # 第一部分：生成节点级检修任务  
    if n_nodes > 0:  
        bus_ids = []  
        for feeder in all_feeders:  
            if n_nodes <= len(all_feeders):  
                # 每条馈线取一个，直到达到n_nodes个为止  
                if len(bus_ids) < n_nodes:  
                    if feeder:  # 若该馈线非空  
                        bus_ids.append(random.choice(feeder))  
            else:  
                # 每条馈线都取一个“基础任务”，多余部分全局随机  
                if feeder:  
                    bus_ids.append(random.choice(feeder))  
        # 如果n_nodes多于馈线数，则补充剩余任务  
        if len(bus_ids) < n_nodes:  
            remaining = list(set(net.bus.index) - set(bus_ids))  
            extra = random.sample(remaining, n_nodes - len(bus_ids))  
            bus_ids.extend(extra) 
        for bus_id in bus_ids:  
            node_name = net.bus.at[bus_id, "name"] if "name" in net.bus.columns else f"bus{bus_id}"  

            # ====== 新增：确定馈线/只遍历本馈线内 =========  
            this_feeder_id = feeder_map.get(bus_id, None)  
            feeder_nodes = set(bus for bus, f in feeder_map.items() if f == this_feeder_id)  
            # 计算全下游（含本bus）受影响bus  
            if feeder_nodes:  
                downstream_buses = identify_downstream_nodes_feeder(net, bus_id, feeder_nodes, [])  
                affected_buses = [bus_id] + downstream_buses  
            else:  
                affected_buses = [bus_id]  

            # 收集与该节点相关的所有设备  
            related_equipment = {  
                "lines": [],  
                "switches": [],  
                "trafos": []  
            }  

            # 查找关联线路  
            if hasattr(net, "line"):  
                for lid, row in net.line.iterrows():  
                    if row["from_bus"] == bus_id or row["to_bus"] == bus_id:  
                        related_equipment["lines"].append(int(lid))  

            # 查找关联开关  
            if hasattr(net, "switch"):  
                for sid, row in net.switch.iterrows():  
                    if row["bus"] == bus_id:  
                        related_equipment["switches"].append(int(sid))  

            # 查找关联变压器  
            if hasattr(net, "trafo"):  
                for tid, row in net.trafo.iterrows():  
                    if row["hv_bus"] == bus_id or row["lv_bus"] == bus_id:  
                        related_equipment["trafos"].append(int(tid))  

            # 只有当至少有一个设备时，才添加任务  
            if sum(len(v) for v in related_equipment.values()) > 0:  
                # 收集受影响负荷  
                affected_loads = []  
                if hasattr(net, 'load'):  
                    for lid, load in net.load.iterrows():  
                        if load['bus'] in affected_buses:  
                            affected_loads.append(int(lid))  

                # 使用改进的转供判定逻辑  
                transfer_info = determine_transfer_need(net, affected_buses, affected_loads)  

                # 判断检修性质 - 开关多为冷备用，节点和线路为检修  
                action = "检修"  
                if len(related_equipment["switches"]) > 0 and not related_equipment["lines"] and not related_equipment["trafos"]:  
                    action = "冷备用"  

                tasks.append({  
                    "task_id": task_id,  
                    "task_type": "node",  
                    "feeder_id": this_feeder_id,              # === 新增字段 ===  
                    "node_id": bus_id,  
                    "node_name": node_name,  
                    "action": action,  
                    "start_time": get_random_datetime(),  
                    "duration_h": random.choice([4, 6, 8]),  
                    "priority": random.randint(1, 3),  
                    "equipment": related_equipment,  
                    "equipment_counts": {k: len(v) for k, v in related_equipment.items()},  
                    "affected_loads": affected_loads,  
                    "affected_buses": affected_buses,         # === 新增，为结构一致  
                    "needs_transfer": transfer_info["needs_transfer"],  
                    "transfer_reason": transfer_info["transfer_reason"] if transfer_info["needs_transfer"] else "",  
                    "total_load_mw": transfer_info.get("total_load_mw", 0),  
                    "desc": f"节点{node_name}({bus_id})整体{action}计划" +   
                            (f"（需转供：{transfer_info['transfer_reason']}）" if transfer_info["needs_transfer"] else "")  
                })  
                task_id += 1  
    
    # 第二部分：生成单独线路检修任务  
    if n_single_lines > 0 and hasattr(net, "line"):  
        # 选择不在节点任务中的线路  
        used_lines = []  
        for t in tasks:  
            if t["task_type"] == "node":  
                used_lines.extend(t["equipment"]["lines"])  
        
        # 从剩余线路中选择  
        available_lines = [l for l in net.line.index if l not in used_lines]  
        if available_lines:  
            line_ids = random.sample(available_lines, min(n_single_lines, len(available_lines)))  
            
            for lid in line_ids:  
                # 识别线路连接的所有负荷节点  
                from_bus = net.line.at[lid, "from_bus"]  
                to_bus   = net.line.at[lid, "to_bus"]  

                # 获得所属馈线编号  
                this_feeder_id = feeder_map.get(from_bus, None)  
                if this_feeder_id is None: this_feeder_id = feeder_map.get(to_bus, None)  
                feeder_nodes = set(bus for bus, f in feeder_map.items() if f == this_feeder_id)  

                # 新下游遍历代码（只在属于本馈线的bus查找受影响bus）  
                disconnected_elements = [('line', lid)]  
                downstream_buses = identify_downstream_nodes_feeder(net, from_bus, feeder_nodes, disconnected_elements)  
                downstream_buses += identify_downstream_nodes_feeder(net, to_bus, feeder_nodes, disconnected_elements)  
                downstream_buses = list(set(downstream_buses))  
                if not downstream_buses:  
                    downstream_buses = [from_bus, to_bus]  
                
                # 收集受影响负荷  
                affected_loads = []  
                if hasattr(net, 'load'):  
                    for load_id, load in net.load.iterrows():  
                        if load['bus'] in downstream_buses:  
                            affected_loads.append(int(load_id))  
                
                # 使用改进的转供判定逻辑  
                transfer_info = determine_transfer_need(net, downstream_buses, affected_loads)  
                
                tasks.append({  
                    "feeder_id": this_feeder_id, 
                    "task_id": task_id,  
                    "task_type": "single",  
                    "element_type": "line",  
                    "element_id": lid,  
                    "action": "检修",  
                    "start_time": get_random_datetime(),  
                    "duration_h": random.choice([4, 6, 8]),  
                    "priority": random.randint(1, 3),  
                    "affected_loads": affected_loads,  
                    "affected_buses": downstream_buses,  
                    "needs_transfer": transfer_info["needs_transfer"],  
                    "transfer_reason": transfer_info["transfer_reason"] if transfer_info["needs_transfer"] else "",  
                    "total_load_mw": transfer_info.get("total_load_mw", 0),  
                    "desc": f"线路#{lid}检修" +   
                            (f"（需转供：{transfer_info['transfer_reason']}）" if transfer_info["needs_transfer"] else "")  
                })  
                task_id += 1  
    
    # 第三部分：生成单独开关检修/冷备用任务  
    if n_single_switches > 0 and hasattr(net, "switch"):  
        used_switches = []  
        for t in tasks:  
            if t["task_type"] == "node":  
                used_switches.extend(t["equipment"]["switches"])  
        available_switches = [s for s in net.switch.index if s not in used_switches]  
        if available_switches:  
            switch_ids = random.sample(available_switches, min(n_single_switches, len(available_switches)))  
            for sid in switch_ids:  
                bus_id = net.switch.at[sid, "bus"]  
                element_type = net.switch.at[sid, "et"]  
                affected_loads = []  
                affected_buses = [bus_id] if bus_id is not None else []  
                this_feeder_id = feeder_map.get(bus_id, None)  
                # ================关键更改如下========================  
                # 线路开关  
                if element_type == 'l' and hasattr(net.switch, 'element'):  
                    line_id = net.switch.at[sid, 'element']  
                    if line_id is not None and line_id in net.line.index:  
                        from_bus = net.line.at[line_id, 'from_bus']  
                        to_bus = net.line.at[line_id, 'to_bus']  
                        # 查from_bus馈线  
                        feeder_id_l = feeder_map.get(from_bus, None)  
                        feeder_nodes_l = set(bus for bus, f in feeder_map.items() if f == feeder_id_l)  
                        if feeder_nodes_l:  
                            disconnected_elements = [('switch', sid)]  
                            downstream_buses = identify_downstream_nodes_feeder(net, from_bus, feeder_nodes_l, disconnected_elements)  
                            downstream_buses += identify_downstream_nodes_feeder(net, to_bus, feeder_nodes_l, disconnected_elements)  
                            affected_buses.extend(list(set(downstream_buses)))  
                        # 记录本任务的馈线为from_bus侧馈线  
                        this_feeder_id = feeder_id_l  
                # 母联/trafo/bus等其他类型可和上面类似  
                # 通用，只要有bus_id的就关联馈线  
                # ================关键更改结束========================  
                affected_buses = list(set(affected_buses))  
                # 查负荷  
                if hasattr(net, 'load'):  
                    for load_id, load in net.load.iterrows():  
                        if load['bus'] in affected_buses:  
                            affected_loads.append(int(load_id))  
                is_closed = net.switch.at[sid, "closed"] if "closed" in net.switch.columns else True  
                if is_closed:  
                    transfer_info = determine_transfer_need(net, affected_buses, affected_loads)  
                else:  
                    transfer_info = {"needs_transfer": False, "transfer_reason": "开关已断开"}  
                tasks.append({  
                    "task_id": task_id,  
                    "task_type": "single",  
                    "element_type": "switch",  
                    "element_id": sid,  
                    "feeder_id": this_feeder_id,            # === 新增字段 ===  
                    "switch_type": element_type,  
                    "action": "冷备用",  
                    "start_time": get_random_datetime(),  
                    "duration_h": random.choice([2, 4, 6]),  
                    "priority": random.randint(1, 3),  
                    "affected_loads": affected_loads,  
                    "affected_buses": affected_buses,  
                    "needs_transfer": transfer_info["needs_transfer"],  
                    "transfer_reason": transfer_info["transfer_reason"] if transfer_info["needs_transfer"] else "",  
                    "total_load_mw": transfer_info.get("total_load_mw", 0),  
                    "desc": f"开关#{sid}转冷备用" +  
                            (f"（需转供：{transfer_info['transfer_reason']}）" if transfer_info["needs_transfer"] else "")  
                })  
                task_id += 1  
    
    # 第四部分：生成单独变压器检修/冷备用任务  
    if n_single_trafos > 0 and hasattr(net, "trafo"):  
        used_trafos = []  
        for t in tasks:  
            if t["task_type"] == "node":  
                used_trafos.extend(t["equipment"]["trafos"])  
        available_trafos = [t for t in net.trafo.index if t not in used_trafos]  
        if available_trafos:  
            trafo_ids = random.sample(available_trafos, min(n_single_trafos, len(available_trafos)))  
            for tid in trafo_ids:  
                hv_bus = net.trafo.at[tid, "hv_bus"]  
                lv_bus = net.trafo.at[tid, "lv_bus"]  
                # ==== 查所属馈线并限定只遍历本馈线 ====  
                this_feeder_id = feeder_map.get(lv_bus, None)  
                feeder_nodes = set(bus for bus, f in feeder_map.items() if f == this_feeder_id)  
                if feeder_nodes:  
                    disconnected_elements = [('trafo', tid)]  
                    downstream_buses = identify_downstream_nodes_feeder(net, lv_bus, feeder_nodes, disconnected_elements)  
                    affected_buses = [lv_bus] + downstream_buses  
                else:  
                    affected_buses = [lv_bus]  
                # 查受影响负荷  
                affected_loads = []  
                if hasattr(net, 'load'):  
                    for load_id, load in net.load.iterrows():  
                        if load['bus'] in affected_buses:  
                            affected_loads.append(int(load_id))  
                transfer_info = determine_transfer_need(net, affected_buses, affected_loads)  
                priority = 1 if transfer_info["needs_transfer"] and "一级" in transfer_info.get("transfer_reason", "") else 2  
                tasks.append({  
                    "task_id": task_id,  
                    "task_type": "single",  
                    "element_type": "trafo",  
                    "element_id": tid,  
                    "feeder_id": this_feeder_id,                # === 新增字段 ===  
                    "action": "冷备用",  # 变压器多为冷备用  
                    "start_time": get_random_datetime(),  
                    "duration_h": random.choice([6, 8, 12]),  
                    "priority": priority,  
                    "affected_loads": affected_loads,  
                    "affected_buses": affected_buses,  
                    "needs_transfer": transfer_info["needs_transfer"],  
                    "transfer_reason": transfer_info["transfer_reason"] if transfer_info["needs_transfer"] else "",  
                    "total_load_mw": transfer_info.get("total_load_mw", 0),  
                    "desc": f"变压器#{tid}转冷备用" +   
                            (f"（需转供：{transfer_info['transfer_reason']}）" if transfer_info["needs_transfer"] else "")  
                })  
                task_id += 1  

    # 第五部分：检查和解决任务冲突  
    if resolve_conflicts and len(tasks) > 1:  
        # 检测任务冲突  
        conflicts = detect_task_conflicts(tasks, net)  
        
        if conflicts:  
            print(f"检测到{len(conflicts)}个任务冲突，正在解决...")  
            # 解决冲突  
            tasks = resolve_task_conflicts(tasks, conflicts, net)  
            
            # 再次检查冲突  
            remaining_conflicts = detect_task_conflicts(tasks, net)  
            if remaining_conflicts:  
                print(f"自动解决后仍有{len(remaining_conflicts)}个冲突，可能需要手动调整")  
            else:  
                print("所有冲突已解决")  
    
    # 第六部分：电气潮流验证  
    if validate_with_powerflow:  
        try:  
            verification_results = validate_maintenance_plan_with_powerflow(net, tasks)  
            
            # 更新任务，添加验证结果  
            for i, task in enumerate(tasks):  
                if i < len(verification_results):  
                    result = verification_results[i]  
                    task["pf_valid"] = result["is_valid"]  
                    
                    # 如果验证失败，添加失败原因  
                    if not result["is_valid"]:  
                        violations = result["violations"]  
                        if not violations["convergence"]:  
                            task["pf_message"] = "潮流计算不收敛"  
                        elif violations["voltage_violations"]:  
                            task["pf_message"] = f"电压越限：{len(violations['voltage_violations'])}处"  
                        elif violations["loading_violations"]:  
                            task["pf_message"] = f"设备过载：{len(violations['loading_violations'])}处"  
                        
                        # 更新任务描述  
                        task["desc"] += f"（潮流验证：不通过 - {task.get('pf_message', '')}）"  
                    else:  
                        task["desc"] += "（潮流验证：通过）"  
            
            # 统计验证结果  
            valid_count = sum(1 for result in verification_results if result["is_valid"])  
            print(f"潮流验证完成: {valid_count}/{len(verification_results)}个任务通过验证")  
            
        except Exception as e:  
            print(f"潮流验证过程中出错: {str(e)}")  
            # 继续使用未验证的任务  
    

    return tasks  

# 示例用法：  
if __name__ == "__main__":  
    import pandapower.networks  
    net = pandapower.networks.mv_oberrhein()  
    
    print("正在生成检修计划...")  
    plan = generate_realistic_maintenance_plan(  
        net,   
        n_nodes=5,             # 2个节点级任务  
        n_single_lines=5,      # 3个单独线路检修  
        n_single_switches=5,   # 4个单独开关冷备用  
        n_single_trafos=1,     # 1个变压器冷备用  
        start_date="2025-05-01",  
        days_span=7,           # 计划跨越7天  
        seed=123,  
        validate_with_powerflow=False,  # 启用潮流验证  
        resolve_conflicts=False          # 启用冲突解决  
    )  
    
    # 输出结果概要  
    print(f"\n成功生成检修计划，共{len(plan)}个任务:")  
    print(f"- 节点级任务: {sum(1 for t in plan if t['task_type'] == 'node')}个")  
    print(f"- 单线路任务: {sum(1 for t in plan if t['task_type'] == 'single' and t['element_type'] == 'line')}个")  
    print(f"- 单开关任务: {sum(1 for t in plan if t['task_type'] == 'single' and t['element_type'] == 'switch')}个")  
    print(f"- 单变压器任务: {sum(1 for t in plan if t['task_type'] == 'single' and t['element_type'] == 'trafo')}个")  
    
    # 需要转供的任务  
    transfer_tasks = [t for t in plan if t.get("needs_transfer", False)]  
    print(f"需要转供的任务: {len(transfer_tasks)}个")  
    
    # 潮流验证失败的任务  
    invalid_tasks = [t for t in plan if t.get("pf_valid", True) == False]  
    if invalid_tasks:  
        print(f"潮流验证未通过的任务: {len(invalid_tasks)}个")  
        for t in invalid_tasks:  
            print(f"  - 任务{t['task_id']}: {t['desc']}")  
    
    # 打印节点级任务  
    print("\n===节点级检修任务===")  
    for task in plan:  
        if task["task_type"] == "node":  
            print(f"任务ID: {task['task_id']}, 节点: {task['node_name']}({task['node_id']}), 操作: {task['action']}")  
            print(f"相关设备: 线路{task['equipment_counts']['lines']}条, " +  
                 f"开关{task['equipment_counts']['switches']}个, " +  
                 f"变压器{task['equipment_counts']['trafos']}台")  
            print(f"负荷转供需求: {'需要' if task['needs_transfer'] else '不需要'}")  
            if task['needs_transfer']:  
                print(f"转供原因: {task['transfer_reason']}")  
                print(f"影响负荷总量: {task.get('total_load_mw', 0):.2f}MW")  
            
            if 'pf_valid' in task:  
                print(f"潮流验证: {'通过' if task['pf_valid'] else '不通过'}")  
                if not task.get('pf_valid', True):  
                    print(f"验证信息: {task.get('pf_message', '')}")  
                    
            print(f"详情: {task['desc']}")  
            print("---")  
    
    # 打印单设备任务  
    print("\n===单设备检修任务===")  
    for task in plan:  
        if task["task_type"] == "single":  
            print(f"任务ID: {task['task_id']}, " +  
                 f"类型: {task['element_type']}, " +  
                 f"设备ID: {task['element_id']}")  
            print(f"操作: {task['action']}, 时间: {task['start_time']}, 持续: {task['duration_h']}小时")  
            print(f"负荷转供需求: {'需要' if task['needs_transfer'] else '不需要'}")  
            if task['needs_transfer']:  
                print(f"转供原因: {task['transfer_reason']}")  
                print(f"影响负荷总量: {task.get('total_load_mw', 0):.2f}MW")  
            
            if 'pf_valid' in task:  
                print(f"潮流验证: {'通过' if task['pf_valid'] else '不通过'}")  
                if not task.get('pf_valid', True):  
                    print(f"验证信息: {task.get('pf_message', '')}")  

                    
            print(f"详情: {task['desc']}")  
            print("---")  
        # 生成Excel格式的检修计划报表  

    def create_excel_report(tasks, filename="maintenance_plan.xlsx"):  
        """将任务转换为Excel格式的检修计划"""  
        # 构建表格数据  
        data = []  
        for t in tasks:  
            if t["task_type"] == "node":  
                equipment_str = f"线路:{','.join(map(str, t['equipment']['lines']))} " + \
                              f"开关:{','.join(map(str, t['equipment']['switches']))} " + \
                              f"变压器:{','.join(map(str, t['equipment']['trafos']))}"  
                row = {  
                    "任务ID": t["task_id"],  
                    "所属馈线": t.get("feeder_id", ""), 
                    "任务类型": "节点级任务",  
                    "设备类型": "综合",  
                    "设备ID": t["node_id"],  
                    "设备名称": t["node_name"],  
                    "所属节点": t["node_id"],  
                    "操作性质": t["action"],  
                    "开始时间": t["start_time"],  
                    "持续时间(小时)": t["duration_h"],  
                    "优先级": t["priority"],  
                    "相关设备": equipment_str,  
                    "受影响负荷": ",".join(map(str, t["affected_loads"])),  
                    "需要转供": "是" if t["needs_transfer"] else "否",  
                    "转供原因": t.get("transfer_reason", ""),  
                    "影响负荷总量(MW)": t.get("total_load_mw", 0),  
                    "潮流验证": "通过" if t.get("pf_valid", True) else "不通过",  
                    "验证信息": t.get("pf_message", ""),  
                    "任务描述": t["desc"]  
                }  
            else:  # 单设备任务  
                row = {  
                    "任务ID": t["task_id"],  
                    "所属馈线": t.get("feeder_id", ""), 
                    "任务类型": "单设备任务",  
                    "设备类型": t["element_type"],  
                    "设备ID": t["element_id"],  
                    "设备名称": f"{t['element_type']}_{t['element_id']}",  
                    "所属节点": ",".join(map(str, t.get("affected_buses", []))),  
                    "操作性质": t["action"],  
                    "开始时间": t["start_time"],  
                    "持续时间(小时)": t["duration_h"],  
                    "优先级": t["priority"],  
                    "相关设备": "",  
                    "受影响负荷": ",".join(map(str, t["affected_loads"])),  
                    "需要转供": "是" if t["needs_transfer"] else "否",  
                    "转供原因": t.get("transfer_reason", ""),  
                    "影响负荷总量(MW)": t.get("total_load_mw", 0),  
                    "潮流验证": "通过" if t.get("pf_valid", True) else "不通过",  
                    "验证信息": t.get("pf_message", ""),  
                    "任务描述": t["desc"]  
                }  
            data.append(row)  
        
        # 创建DataFrame并保存为Excel  
        df = pd.DataFrame(data)  
        df.to_excel(filename, index=False)  
        return filename  
    # 输出Excel报表  
    excel_file = f"maintenance_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"  
    create_excel_report(plan, excel_file)  
    print(f"\n检修计划Excel报表已生成: {excel_file}")            