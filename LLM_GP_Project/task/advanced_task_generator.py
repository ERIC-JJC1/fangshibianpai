import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import copy
import pandapower as pp
import pandapower.topology as top
import networkx as nx

def traverse_feeder(G, start_bus, first_neighbor, exclude_buses):
    """遍历馈线以确定其包含的节点"""
    visited = set([start_bus, first_neighbor])
    stack = [first_neighbor]
    while stack:
        current = stack.pop()
        for nb in G.neighbors(current):
            if nb not in visited and nb not in exclude_buses:
                visited.add(nb)
                stack.append(nb)
    return sorted(visited)

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

def generate_realistic_maintenance_plan(
    net,
    feeder_bus_sets,
    feeder_map,
    n_nodes=5,         # 节点级任务数量
    n_single_lines=5,  # 单独线路任务数量
    n_single_switches=5, # 单独开关任务数量
    n_single_trafos=1,  # 单独变压器任务数量
    start_date="2025-06-01",
    days_span=7,       # 计划跨度天数
    seed=None,
):
    """
    生成真实可执行的检修计划，包括:
    1. 节点级检修任务：每个任务包含节点相关的所有设备
    2. 单设备检修任务：针对单个线路、开关或变压器的检修
    3. 标记负荷转供需求：对于导致负荷失电的检修，标记需要转供
    """
    random.seed(seed)
    tasks = []
    task_id = 1
    
    # 构建馈线列表
    all_feeders = []
    for fid in sorted(feeder_bus_sets.keys()):
        all_feeders.append(list(feeder_bus_sets[fid]))
    
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
                # 每条馈线都取一个"基础任务"，多余部分全局随机
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
                    "feeder_id": this_feeder_id,
                    "node_id": bus_id,
                    "node_name": node_name,
                    "action": action,
                    "start_time": get_random_datetime(),
                    "duration_h": random.choice([4, 6, 8]),
                    "priority": random.randint(1, 3),
                    "equipment": related_equipment,
                    "equipment_counts": {k: len(v) for k, v in related_equipment.items()},
                    "affected_loads": affected_loads,
                    "affected_buses": affected_buses,
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
                if this_feeder_id is None: 
                    this_feeder_id = feeder_map.get(to_bus, None)
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
                    "feeder_id": this_feeder_id,
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
                    "feeder_id": this_feeder_id,
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

    return tasks