import numpy as np
from datetime import datetime, timedelta
import os
import pandas as pd   
import copy
import pandapower as pp  
import pandapower.networks as pn  
import pandapower.plotting as ppp  
import networkx as nx    
from pandapower.topology import create_nxgraph
import re
import itertools


# ========== 直接定义基类 ==========
class MultiAgentEnv:
    """多智能体环境基类"""
    def __init__(self):
        pass
    def step(self, actions):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def get_obs(self):
        raise NotImplementedError
    def get_state(self):
        raise NotImplementedError
    def get_avail_actions(self):
        return None
    def get_env_info(self):
        raise NotImplementedError
    def render(self):
        pass
    def close(self):
        pass

# 设定馈线之间的联络开关
inter_feeder_switch_map = {
    1: [  
        # 馈线1与馈线2的联络（66号线）
        {'with_feeder': 2, 'switch_pair': [106, 107], 'line': 66, 'bus1': 54, 'bus2': 147}
    ],
    2: [
        {'with_feeder': 1, 'switch_pair': [106, 107], 'line': 66, 'bus1': 147, 'bus2': 54},
        {'with_feeder': 3, 'switch_pair': [33, 34], 'line': 23, 'bus1': 195, 'bus2': 132},
        {'with_feeder': 3, 'switch_pair': [143, 144], 'line': 88, 'bus1': 236, 'bus2': 223},
        {'with_feeder': 4, 'switch_pair': [47, 48],  'line': 31, 'bus1': 31, 'bus2': 190}
    ],
    3: [
        {'with_feeder': 2, 'switch_pair': [33, 34], 'line': 23, 'bus1': 132, 'bus2': 195},
        {'with_feeder': 2, 'switch_pair': [143, 144], 'line': 88, 'bus1': 223, 'bus2': 236},
        {'with_feeder': 4, 'switch_pair': [264, 265], 'line': 162, 'bus1': 39, 'bus2': 80},
        {'with_feeder': 4, 'switch_pair': [310, 311], 'line': 188, 'bus1': 35, 'bus2': 45},
    ],
    4: [
        {'with_feeder': 2, 'switch_pair': [47, 48], 'line': 31, 'bus1': 190, 'bus2': 31},
        {'with_feeder': 3, 'switch_pair': [264, 265], 'line': 162, 'bus1': 80, 'bus2': 39},
        {'with_feeder': 3, 'switch_pair': [310, 311], 'line': 188, 'bus1': 45, 'bus2': 35},
    ]
}
# 提取所有成对联络开关
switch_pair_list = []
for feed, links in inter_feeder_switch_map.items():
    for link in links:
        switch_pair_list.append(tuple(link['switch_pair']))  # (A, B)

#设定转供方案

def get_feeder_transfer_schemes(feeder_id):
    """
    生成馈线转供方案，去除"不转供"选项
    """
    candidate_lines = inter_feeder_switch_map[feeder_id]  # 本馈线所有联络
    n = len(candidate_lines)
    transfer_list = []
    
    idx = 0  # ✅ 从0开始编号
    # 生成1~n条联络线的所有组合
    for k in range(1, n+1):
        for combo in itertools.combinations(candidate_lines, k):
            switch_ops = []
            desc_items = []
            for link in combo:
                desc_items.append(f"{link['with_feeder']} via line {link['line']}")
                switch_ops.append({'switch_id': link['switch_pair'][0], 'closed': True})
                switch_ops.append({'switch_id': link['switch_pair'][1], 'closed': True})
                
            transfer_scheme = {
                'id': idx,
                'desc': '转供: ' + ', '.join(desc_items),
                'switch_ops': switch_ops,
                'target_feeders': [link['with_feeder'] for link in combo]  # ✅ 添加目标馈线信息
            }
            transfer_list.append(transfer_scheme)
            idx += 1
            
    return transfer_list

def parse_element(element_name):
    en = element_name.lower().replace(" ", "")
    if en.startswith('bus'):
        element_type = 'bus'
    elif en.startswith('line'):
        element_type = 'line'
    elif en.startswith('switch'):
        element_type = 'switch'
    elif en.startswith('trafo'):
        element_type = 'trafo'
    else:
        return "unknown", None

    matches = re.findall(r'\d+', en)
    element_id = int(matches[0]) if matches else None
    return element_type, element_id
    
def read_tasks_from_excel(file_path): #读取excel
    df = pd.read_excel(file_path)
    tasks = []
    for idx, row in df.iterrows():
        if pd.isnull(row['所属馈线']) or pd.isnull(row['设备名称']) or pd.isnull(row['开始时间']) or pd.isnull(row['持续时间(小时)']) or pd.isnull(row['优先级']):
            continue

        feeder_id = int(row['所属馈线'])          # 馈线编号，需与环境的馈线ID对齐
        region_id = feeder_id - 1
        element_name = str(row['设备名称'])       # 元件名称
        start_time = pd.to_datetime(row['开始时间'])
        duration = int(row['持续时间(小时)'])
        priority = int(row['优先级'])           # 优先级：1低，2中，3高
        
        element_type, element_id = parse_element(element_name)
        
        # 构造任务字典
        task = {
            'feeder_id': feeder_id,  # 馈线ID
            'region_id': region_id,
            'element_type': element_type,
            'element_id': element_id,
            'element_name': element_name,
            'start_time': start_time.strftime("%Y-%m-%d %H:%M"),
            'duration': duration,
            'priority': priority,
            'status': 'unassigned',  # 初始未分配
        }
        tasks.append(task)
    return tasks


#============识别馈线===========================================
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

#--------------------------分配馈线--------------------------
def build_region_assets(net, feeders_bus_sets):
    region_buses = {i: list(feeders_bus_sets[i]) for i in range(4)}
    region_lines = {i: [] for i in range(4)}
    region_switches = {i: [] for i in range(4)}
    region_trafos = {i: [] for i in range(4)}
    # 分配 line
    for idx, line in net.line.iterrows():
        for agent_id, feeder_set in enumerate(feeders_bus_sets):
            if line["from_bus"] in feeder_set and line["to_bus"] in feeder_set:
                region_lines[agent_id].append(idx)
                break
    # 分配 switch
    for idx, switch in net.switch.iterrows():
        for agent_id, feeder_set in enumerate(feeders_bus_sets):
            if switch["bus"] in feeder_set:
                region_switches[agent_id].append(idx)
                break
    # 分配 trafo
    for idx, trafo in net.trafo.iterrows():
        for agent_id, feeder_set in enumerate(feeders_bus_sets):
            if trafo["hv_bus"] in feeder_set or trafo["lv_bus"] in feeder_set:
                region_trafos[agent_id].append(idx)
                break
    # 构建区域资产字典
    region_assets = {
        i: {
            "buses": region_buses[i],
            "lines": region_lines[i],
            "switches": region_switches[i],
            "trafos": region_trafos[i],
        } for i in range(4)
    }
    for region_id, assets in region_assets.items():
        print(f"区域 {region_id}:")
        print(f"  母线: {assets['buses']}")
        print(f"  线路索引: {assets['lines']}")
        print(f"  开关索引: {assets['switches']}")
        print(f"  变压器索引: {assets['trafos']}")
        print()  # 添加空行以区分区域输出

    return region_assets



class GridMaintenanceEnv(MultiAgentEnv):
        
    def assign_tasks_to_agents(self):
        element_mapping = {
            'bus': 'buses',
            'line': 'lines', 
            'switch': 'switches',
            'trafo': 'trafos'
        }
        
        unassigned_tasks = []
        """根据element_id将任务分配给智能体"""
        for task in self.tasks_all:
            element_type = task['element_type']
            element_id = task['element_id']
            assigned = False
            
            if element_type in element_mapping:
                asset_key = element_mapping[element_type]
                for agent_id in range(self.agent_num):
                    if element_id in self.region_assets[agent_id][asset_key]:
                        task['region_id'] = agent_id
                        assigned = True
                    break
            
            if not assigned:
                feeder_id = task.get('feeder_id')
                if feeder_id and 1 <= feeder_id <= 4:
                    task['region_id'] = feeder_id - 1
                    assigned = True
                    print(f"[任务分配] {task['element_name']} 根据feeder_id {feeder_id} 分配给 Agent {feeder_id-1}")
            
            if not assigned:
                task['region_id'] = -1
                unassigned_tasks.append(task)
                print(f"[警告] 任务 {task['element_name']} 无法分配到任何智能体")

        return {
            'total_tasks': len(self.tasks_all),
            'assigned_tasks': len(self.tasks_all) - len(unassigned_tasks),
            'unassigned_tasks': len(unassigned_tasks)
        }        
        

    def get_compatible_transfer_schemes(self, agent_id, time_slot, duration):
        """
        获取与其他智能体任务兼容的转供方案
        
        Args:
            agent_id: 当前智能体ID
            time_slot: 任务开始时间槽
            duration: 任务持续时间
        
        Returns:
            List[int]: 可用的转供方案索引列表
        """
        feeder_id = agent_id + 1
        my_transfer_schemes = get_feeder_transfer_schemes(feeder_id)
        
        # 获取时间重叠的其他智能体已分配任务
        conflicting_tasks = []
        for other_agent_id in range(self.n_agents):
            if other_agent_id == agent_id:
                continue
                
            other_tasks = [t for t in self.tasks 
                        if t['region_id'] == other_agent_id and t['status'] == 'assigned']
            
            for task in other_tasks:
                task_start = task['assigned_time_idx']
                task_end = task_start + task['duration']
                
                # 检查时间重叠
                my_start = time_slot
                my_end = time_slot + duration
                
                if not (my_end <= task_start or my_start >= task_end):  # 有重叠
                    conflicting_tasks.append({
                        'agent_id': other_agent_id,
                        'feeder_id': other_agent_id + 1,
                        'transfer_idx': task.get('transfer_idx', 0),
                        'task': task
                    })
        
        if not conflicting_tasks:
            # 没有冲突，所有方案都可用
            return list(range(len(my_transfer_schemes)))
        
        # ✅ 核心逻辑：找兼容的转供方案
        compatible_schemes = []
        
        for scheme_idx, my_scheme in enumerate(my_transfer_schemes):
            is_compatible = True
            my_target_feeders = set(my_scheme['target_feeders'])
            
            for conflict in conflicting_tasks:
                conflict_feeder = conflict['feeder_id']
                conflict_transfer_idx = conflict['transfer_idx']
                
                # 获取冲突任务的转供方案
                conflict_schemes = get_feeder_transfer_schemes(conflict_feeder)
                if conflict_transfer_idx < len(conflict_schemes):
                    conflict_scheme = conflict_schemes[conflict_transfer_idx]
                    conflict_targets = set(conflict_scheme['target_feeders'])
                    
                    # ✅ 兼容性检查：如果我要转供到馈线X，且馈线X有冲突任务，
                    # 那么馈线X的任务必须转供回到我的馈线
                    if conflict_feeder in my_target_feeders:
                        if feeder_id not in conflict_targets:
                            is_compatible = False
                            break
            
            if is_compatible:
                compatible_schemes.append(scheme_idx)
        
        return compatible_schemes      

    def __init__(self, task_file=None, initial_tasks=None, forecast_loads=None, 
                horizon_days=7, window_size=3, load_curve_file=None, **kwargs):
        super().__init__()

        # ✅ 添加优先级时间窗口配置
        self.priority_time_windows = {
            3: 0,    # 高优先级：时间固定（0天窗口）
            2: 3,    # 中优先级：3天窗口  
            1: 7     # 低优先级：7天窗口
        }

        # 1. 初始化网络
        net = pp.networks.mv_oberrhein(
            scenario='generation',
            cosphi_load=0.98,
            cosphi_pv=1.0,
            include_substations=False,
            separation_by_sub=False
        )
        self.net = net
        self.initial_net = copy.deepcopy(net)

        # 2. 加载任务数据（优先级：task_file > initial_tasks > 空）
        if task_file is not None:
            print(f"[环境初始化] 从文件加载任务: {task_file}")
            if os.path.exists(task_file):
                self.tasks_all = read_tasks_from_excel(task_file)
                print(f"[环境初始化] 成功加载 {len(self.tasks_all)} 个任务")
            else:
                print(f"[警告] 任务文件不存在: {task_file}")
                self.tasks_all = []
        elif initial_tasks is not None:
            print(f"[环境初始化] 使用传入的任务数据")
            self.tasks_all = [dict(t) for t in initial_tasks]
        else:
            print(f"[警告] 未提供任务数据")
            self.tasks_all = []

        # 3. 为任务添加转供方案
        for task in self.tasks_all:
            feeder_id = task['feeder_id']
            transfer_options = get_feeder_transfer_schemes(feeder_id)
            task['transfer_options'] = transfer_options
            task['transfer_mask'] = np.ones(len(transfer_options), dtype=np.int_)
            # 打印每个任务及其转供方案
            print(f"任务: {task['element_name']}, 馈线ID: {feeder_id}")
            print(f"  转供方案: {len(transfer_options)} 个选项")
            for option in transfer_options:
                print(f"  - 方案ID: {option['id']}, 描述: {option['desc']}")
            print()  # 添加空行以区分任务之间的输出



        # 6. 智能体设置
        self.agent_num = 4
        self.agents = list(range(self.agent_num))
        self.n_agents = self.agent_num  # 统一属性名
        
        # 7. 时间设置
        self.forecast_loads = forecast_loads
        self.horizon = horizon_days
        self.window_size = window_size
        self.num_simulation_days = 7
        self.hours = list(range(6, 19))  # 6-18点，共13小时
        self.time_slots = self.num_simulation_days * len(self.hours)  # 91
        self.episode_limit = 24 * self.horizon
        self.load_idx_order = list(self.net.load.index)
        
        # ✅ 3. 先计算仿真开始时间（用于时间窗口计算）
        if self.tasks_all:
            start_times = [t["start_time"] for t in self.tasks_all]
            earliest_time = min(start_times)
            self.simulation_start_time = datetime.strptime(earliest_time, "%Y-%m-%d %H:%M")
        else:
            self.simulation_start_time = datetime.strptime("2025-05-01 06:00", "%Y-%m-%d %H:%M")

        # ✅ 4. 为任务添加转供方案和时间窗口
        for task in self.tasks_all:
            feeder_id = task['feeder_id']
            transfer_options = get_feeder_transfer_schemes(feeder_id)
            task['transfer_options'] = transfer_options
            task['transfer_mask'] = np.ones(len(transfer_options), dtype=np.int_)
            
            # 打印任务信息
            print(f"任务: {task['element_name']}, 馈线ID: {feeder_id}, 优先级: {task['priority']}")
            print(f"  转供方案: {len(transfer_options)} 个选项")

        # ✅ 5. 计算每个任务的时间窗口
        self._calculate_task_time_windows()


        # 5. 网络拓扑和区域划分（保持原有逻辑）
        important_buses = [167, 273, 244, 65, 148, 216, 227]  
        net.bus["important"] = False  
        net.bus.loc[important_buses, "important"] = True
        G = create_nxgraph(net)
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

        feeders_bus_sets = [set(feeder1), set(feeder2), set(feeder3), set(feeder4)]
        self.region_assets = build_region_assets(net, feeders_bus_sets)

        # 4. 加载负荷曲线
        if load_curve_file is not None:
            print(f"[环境初始化] 加载负荷曲线: {load_curve_file}")
            if os.path.exists(load_curve_file):
                self.loads_curve = pd.read_csv(load_curve_file)
                self.loads_curve = self.loads_curve.set_index("load_idx")
                print(f"[环境初始化] 成功加载 {len(self.loads_curve)} 个负荷曲线")
            else:
                print(f"[警告] 负荷曲线文件不存在: {load_curve_file}")
                self.loads_curve = None
        else:
            self.loads_curve = None

        # 8. 任务区域分配
        self.assign_tasks_to_agents()   

        # 9. 重置环境
        self.reset()

        print("==== 任务-智能体归属/Sanity Check ====")
        self.print_task_agent_assignment()
        self.show_agent_tasks()
        print("==== 抽查负荷赋值精确性 ====")
        self.quick_check_all_loads([0, 5, 12, 20, 50, 90])

    # ✅ 新增：时间窗口计算方法
    def _calculate_task_time_windows(self):
        """根据优先级计算每个任务的可选时间窗口"""
        print("\n=== 计算任务时间窗口 ===")
        
        for task in self.tasks_all:
            priority = task['priority']
            original_time = datetime.strptime(task['start_time'], "%Y-%m-%d %H:%M")
            
            if priority == 3:  # 高优先级：时间固定
                fixed_slot = self._datetime_to_time_slot(original_time)
                if fixed_slot >= 0:
                    task['allowed_time_slots'] = [fixed_slot]
                else:
                    # 如果原始时间不在工作时间内，找最近的工作时间
                    task['allowed_time_slots'] = [self._find_nearest_work_time_slot(original_time)]
                task['time_flexibility'] = 'fixed'
                
            elif priority == 2:  # 中优先级：3天窗口
                window_days = self.priority_time_windows[2]
                start_time = original_time
                end_time = original_time + timedelta(days=window_days)
                task['allowed_time_slots'] = self._get_time_slots_in_range(start_time, end_time)
                task['time_flexibility'] = 'medium'
                
            else:  # 低优先级：7天窗口
                window_days = self.priority_time_windows[1]
                start_time = original_time
                end_time = original_time + timedelta(days=window_days)
                task['allowed_time_slots'] = self._get_time_slots_in_range(start_time, end_time)
                task['time_flexibility'] = 'high'
            
            print(f"任务 {task['element_name']} (优先级{priority}): "
                  f"可选时间槽 {len(task['allowed_time_slots'])} 个 "
                  f"({task['time_flexibility']})")

    def _datetime_to_time_slot(self, dt):
        """将datetime转换为时间槽索引"""
        # 计算相对于仿真开始时间的天数差
        days_diff = (dt.date() - self.simulation_start_time.date()).days
        
        if days_diff < 0 or days_diff >= self.num_simulation_days:
            return -1  # 超出仿真范围
        
        hour = dt.hour
        if hour < 6 or hour > 18:
            return -1  # 超出工作时间
        
        hour_idx = hour - 6  # 转换为0-12的索引
        time_slot = days_diff * 13 + hour_idx
        
        return time_slot if time_slot < self.time_slots else -1

    def _find_nearest_work_time_slot(self, dt):
        """找到最近的工作时间槽"""
        # 如果时间太早，使用当天6点
        if dt.hour < 6:
            nearest_dt = dt.replace(hour=6, minute=0, second=0)
        # 如果时间太晚，使用第二天6点
        elif dt.hour > 18:
            nearest_dt = (dt + timedelta(days=1)).replace(hour=6, minute=0, second=0)
        else:
            nearest_dt = dt
        
        slot = self._datetime_to_time_slot(nearest_dt)
        return max(0, min(slot, self.time_slots - 1))

    def _get_time_slots_in_range(self, start_time, end_time):
        """获取时间范围内的所有工作时间槽"""
        slots = []
        current_time = start_time
        
        while current_time <= end_time:
            # 只考虑工作时间6-18点
            for hour in range(6, 19):
                work_time = current_time.replace(hour=hour, minute=0, second=0)
                if work_time <= end_time:
                    slot = self._datetime_to_time_slot(work_time)
                    if slot >= 0 and slot not in slots:
                        slots.append(slot)
            
            current_time += timedelta(days=1)
            if current_time.date() >= self.simulation_start_time.date() + timedelta(days=self.num_simulation_days):
                break
        
        return sorted(slots)

    # ✅ 修改：基于优先级的任务选择
    def _get_next_unassigned_task(self, agent_id):
        """按优先级获取下一个未分配的任务"""
        unassigned_tasks = [
            t for t in self.tasks 
            if t['region_id'] == agent_id and t['status'] == 'unassigned'
        ]
        
        if not unassigned_tasks:
            return None
        
        # 按优先级排序：高优先级 -> 低优先级 -> 早开始时间
        return sorted(unassigned_tasks, key=lambda x: (
            -x['priority'],  # 优先级高的先分配（3->2->1）
            x['start_time']  # 同优先级按开始时间排序
        ))[0]
    
    def check_switch_pairs_consistency(self, auto_fix=True, debug=False):  
        for s1, s2 in switch_pair_list:  
            state1 = self.net.switch.at[s1, 'closed']  
            state2 = self.net.switch.at[s2, 'closed']  
            if state1 != state2:  
                if debug:  
                    print(f"[严重警告] 联络开关对 {s1}-{s2} 状态不同步： {state1}, {state2}")  
            if auto_fix:  
                self.net.switch.at[s1, 'closed'] = True  
                self.net.switch.at[s2, 'closed'] = True  
                if debug:  
                    print(f"[auto-fix] 已自动将 {s1}-{s2} 统一闭合")

    def time_slot_to_day_hour(self, time_slot):
        """
        将时间槽编码转换为 (day, hour)
        time_slot: 0-90 (7天*13小时)
        返回: (day_idx, hour_idx) 其中hour_idx是工作时间内的索引0-12
        """
        day_idx = time_slot // 13  # 第几天 (0-6)
        hour_idx = time_slot % 13  # 当天第几个工作时间段 (0-12)
        return day_idx, hour_idx
    
    def day_hour_to_time_slot(self, day_idx, hour):
        """
        将 (day, hour) 转换为时间槽编码
        day_idx: 0-6
        hour: 6-18
        返回: time_slot (0-90)
        """
        if hour < 6 or hour > 18:
            return -1  # 超出工作时间
        hour_offset = hour - 6
        time_slot = day_idx * 13 + hour_offset
        return time_slot if time_slot < self.time_slots else -1
    
    def get_time_info(self, time_slot):
        """获取时间槽的详细信息"""
        day_idx, hour = self.time_slot_to_day_hour(time_slot)
        return {
            'time_slot': time_slot,
            'day_idx': day_idx,
            'day_name': f'Day{day_idx+1}',
            'hour': hour,
            'time_str': f'Day{day_idx+1}-{hour:02d}:00'
        }    

    def get_avail_actions(self):
        """返回基于优先级时间窗口的可用动作掩码"""
        avail_actions = []
        
        for agent_id in range(self.n_agents):
            # 获取下一个未分配任务（按优先级排序）
            next_task = self._get_next_unassigned_task(agent_id)
            
            if next_task is None:
                # 没有未分配任务，所有动作都不可用
                agent_avail = np.zeros(self.get_total_actions(), dtype=np.int32)
            else:
                # 为这个任务生成基于优先级的可用动作
                agent_avail = self._generate_priority_based_actions(agent_id, next_task)
            
            avail_actions.append(agent_avail)
        
        return avail_actions
    
    def _generate_priority_based_actions(self, agent_id, task):
        """为特定任务生成基于优先级的可用动作"""
        agent_avail = np.zeros(self.get_total_actions(), dtype=np.int32)
        
        allowed_time_slots = task['allowed_time_slots']
        duration = task['duration']
        
        print(f"[动作生成] Agent {agent_id}, 任务 {task['element_name']} "
              f"(优先级{task['priority']}): {len(allowed_time_slots)} 个可选时间槽")
        
        for time_slot in allowed_time_slots:
            # 检查时间槽有效性（持续时间不超出边界）
            if not self._is_valid_time_slot(time_slot, duration):
                continue
            
            # 获取兼容的转供方案（考虑其他智能体的冲突）
            compatible_transfers = self.get_compatible_transfer_schemes(
                agent_id, time_slot, duration
            )
            
            # 启用兼容的动作
            for transfer_idx in compatible_transfers:
                action_id = self.encode_action(time_slot, transfer_idx)
                if action_id < len(agent_avail):
                    agent_avail[action_id] = 1
        
        active_actions = np.sum(agent_avail)
        print(f"[动作生成] Agent {agent_id} 可用动作数: {active_actions}")
        
        return agent_avail

    def _is_valid_time_slot(self, time_slot, duration):
        """
        检查时间槽是否有效（工作时间约束等）
        """
        if time_slot + duration > self.time_slots:
            return False
            
        # 检查是否在工作时间内
        for t in range(time_slot, time_slot + duration):
            day_idx, hour_idx = self.time_slot_to_day_hour(t)
            actual_hour = 6 + hour_idx
            if actual_hour < 6 or actual_hour > 18:  # 超出工作时间
                return False
                
        return True

    def reset(self):
        """重置环境"""
        # 检查是否有任务数据
        if not self.tasks_all:
            print("[警告] 没有任务数据，使用默认时间设置")
            self.current_time = datetime.strptime("2025-05-01 06:00", "%Y-%m-%d %H:%M")
            self.tasks = []
        else:
            # 获取最早任务开始时间
            start_times = [t["start_time"] for t in self.tasks_all]
            earliest_time = min(start_times)
            self.current_time = datetime.strptime(earliest_time, "%Y-%m-%d %H:%M")
            
            # 深拷贝任务列表
            self.tasks = [dict(task) for task in self.tasks_all]
            
            # 确保每个任务都有转供方案
            for task in self.tasks:
                if "transfer_options" not in task:
                    feeder_id = task.get("feeder_id", None)
                    if feeder_id is not None:
                        transfer_options = get_feeder_transfer_schemes(feeder_id)
                        task['transfer_options'] = transfer_options
                        task['transfer_mask'] = np.ones(len(transfer_options), dtype=np.int_)
                    else:
                        print("[警告] 任务缺少feeder_id，无法生成转供方案", task)

        # 重置其他状态
        self.steps = 0
        
        print(f"[环境重置] 开始时间: {self.current_time}, 任务数: {len(self.tasks)}")
        
        # ✅ 确保所有返回值都不是None
        obs = self.get_obs()  # 应该返回有效的观测列表
        state = self.get_state()  # 应该返回有效的状态
        avail_actions = self.get_avail_actions()  # 应该返回有效的可用动作
        
        # 🔍 添加调试信息
        print(f"[DEBUG reset] obs类型: {type(obs)}, 长度: {len(obs) if obs else 'None'}")
        print(f"[DEBUG reset] state类型: {type(state)}, 形状: {state.shape if hasattr(state, 'shape') else 'None'}")
        print(f"[DEBUG reset] avail_actions类型: {type(avail_actions)}, 长度: {len(avail_actions) if avail_actions else 'None'}")
        
        # ✅ 检查None值
        if obs is None:
            print("[ERROR] reset() 返回的 obs 是 None!")
            obs = [np.zeros(self.obs_shape) for _ in range(self.n_agents)]
        
        if state is None:
            print("[ERROR] reset() 返回的 state 是 None!")
            state = np.zeros(self.state_shape)
        
        if avail_actions is None:
            print("[ERROR] reset() 返回的 avail_actions 是 None!")
            avail_actions = [np.ones(self.n_actions) for _ in range(self.n_agents)]
        
        return obs, state, avail_actions


    #--------------------------------------------------------------
    def deactivate_element(self, task):
        """根据任务类型将指定元件断开/停运"""
        element_type = task.get('element_type')
        element_id = task.get('element_id')
        if element_type == 'bus':
            try:
                self.net.bus.at[element_id, "in_service"] = False
            except Exception as e:
                print(f"[deactivate] 断开bus出错: {element_id}, {e}")
        elif element_type == 'line':
            try:
                self.net.line.at[element_id, "in_service"] = False
            except Exception as e:
                print(f"[deactivate] 断开line出错: {element_id}, {e}")
        elif element_type == 'switch':
            try:
                self.net.switch.at[element_id, "closed"] = False
            except Exception as e:
                print(f"[deactivate] 断开switch出错: {element_id}, {e}")
        elif element_type == 'trafo':
            try:
                self.net.trafo.at[element_id, "in_service"] = False
            except Exception as e:
                print(f"[deactivate] 断开trafo出错: {element_id}, {e}")
        else:
            print(f"[deactivate] 未知元件类型: {element_type}, 不处理")

    ##--------------------------定义转供方案和实际操作开关映射的函数---------------------------------------
    def apply_transfer_scheme(self, task):
        transfer_list = task['transfer_options']
        transfer_idx = task.get('transfer_idx', 0)
        switch_ops = transfer_list[transfer_idx]['switch_ops']
        for op in switch_ops:
            switch_id = op['switch_id']
            # 严格写死：只要目标合闸就保证执行（即每遍都apply，不怕多操作幂等）
            if self.net.switch.at[switch_id, "closed"] != op['closed']:
                self.net.switch.at[switch_id, "closed"] = op['closed']
                # print(f"[apply_transfer_scheme] Switch {switch_id} 状态修改为 {op['closed']}")
            # else:
            #     print(f"[apply_transfer_scheme] Switch {switch_id} 已是期望状态 {op['closed']}，不重复操作")

    #----------------------------------------------------------------------------------------
    def step(self, actions):
        """PyMARL标准格式：actions: List[int], 每个智能体一个整数动作"""
        for agent_id, action_id in enumerate(actions):
            # 获取下一个应该分配的任务（按优先级）
            next_task = self._get_next_unassigned_task(agent_id)
            
            if next_task is None: 
                continue  # 没有任务跳过
            
            # 解码动作
            time_slot, transfer_idx = self.decode_action(action_id)
            # 验证动作是否在该任务的允许范围内
            if time_slot not in next_task['allowed_time_slots']:
                print(f"[警告] Agent {agent_id} 选择的时间槽{time_slot}不在任务{next_task['element_name']}的允许范围内")
                # 使用第一个允许的时间槽作为默认
                if next_task['allowed_time_slots']:
                    time_slot = next_task['allowed_time_slots'][0]
                else:
                    continue
            
            # 边界检查
            feeder_id = agent_id + 1
            transfer_schemes = get_feeder_transfer_schemes(feeder_id)
            if transfer_idx >= len(transfer_schemes):
                transfer_idx = 0
            
            # 分配任务
            next_task['assigned_time_idx'] = time_slot
            next_task['transfer_idx'] = transfer_idx
            next_task['status'] = 'assigned'
            
            print(f"[任务分配] Agent {agent_id}: {next_task['element_name']} "
                  f"(优先级{next_task['priority']}) -> 时间槽{time_slot}, "
                  f"转供方案{transfer_idx}")

        # 检查转供方案冲突
        self._check_transfer_conflicts()

        # 计算奖励和状态
        total_reward = self.calc_total_reward()
        obs = self.get_obs()
        state = self.get_state()
        done = all(task['status'] == 'assigned' for task in self.tasks)
        info = {'episode_done': done}

        # 检查联络开关对状态
        self.check_switch_pairs_consistency(auto_fix=True)

        return obs, total_reward, done, info

    def _check_transfer_conflicts(self):
        """检查转供方案冲突"""
        switch_owner = dict()
        for task in self.tasks:
            if task.get('status') == 'assigned':
                idx = task.get('transfer_idx', 0)
                scheme = task['transfer_options'][idx]
                for op in scheme['switch_ops']:
                    if op['closed']:
                        if op['switch_id'] in switch_owner:
                            switch_owner[op['switch_id']].append(task['element_name'])
                        else:
                            switch_owner[op['switch_id']] = [task['element_name']]
        
        # 报告冲突
        for switch_id, owners in switch_owner.items():
            if len(owners) > 1:
                print(f"[警告] switch_id {switch_id} 同时被多个任务闭合: {owners}")

    def restore_net(self):
        # 恢复网络初始值
        # 示例：如果你保存了初始状态，可以在这里做 deep copy
        self.net = copy.deepcopy(self.initial_net)
        pass
        #动作的奖励函数计算#

#----------------------------------检查任务的函数--------------------------
    def print_task_agent_assignment(self):
        print("任务ID | 设备名 | 区域AgentID | 区域名 | 起始时间 | 持续h | 状态")
        for i, t in enumerate(self.tasks):
            region = t.get('region_id', -1)
            print(f"{i:6d} | {t['element_name']:<12} | {region:9d} | feeder{region+1 if region>=0 else 'NA'} | {t['start_time']} | {t['duration']:2d} | {t['status']}")

    def show_agent_tasks(self):
        for agent_id in self.agents:
            my_tasks = [t for t in self.tasks if t.get('region_id', -1)==agent_id]
            print(f"Agent {agent_id}: 共{len(my_tasks)}个任务")
            for t in my_tasks:
                print(f"    设备: {t['element_name']}, 开始: {t['start_time']}, 时长: {t['duration']}h, 状态: {t['status']}")
#---------------------------------------------------
    def check_load_assignment(self, t, verbose=True):
        """
        检查第 t 小时 net.load['p_mw'] 是否与 loads_curve 对应
        只以 load_idx(即 index)为基准，无需类型
        """
        col = f'hour_{t}'
        if col not in self.loads_curve.columns:
            col = self.loads_curve.columns[-1]

        errors = []
        for idx in self.net.load.index:
            if idx in self.loads_curve.index:
                expected = self.loads_curve.at[idx, col]
                now_val = self.net.load.at[idx, 'p_mw']
                if abs(now_val - expected) > 1e-6:
                    errors.append((idx, now_val, expected))
                    if verbose:
                        name = self.net.load.at[idx, "name"]
                        print(f"[负荷校验] load_idx={idx} ({name}) p_mw={now_val:.4f}，期望={expected:.4f}，不符")
            else:
                if verbose:
                    print(f"[负荷校验] load_idx={idx} 在csv中没找到")

        if not errors and verbose:
            print(f"[负荷校验] 第{t}小时全部负荷赋值正确")
        return errors
#--------------------------------------------------------------------
    def quick_check_all_loads(self, sample_hours=[0, 20, 50]):
        for t in sample_hours:
            self.apply_loads_curve(t)
            errors = self.check_load_assignment(t, verbose=False)
            if errors:
                print(f'第{t}小时发现{len(errors)}处负荷不一致')
            else:
                print(f'第{t}小时负荷校验全部OK')



    def apply_loads_curve(self, time_slot):
        """
        应用负荷曲线，从完整7天数据中选择工作时间段
        time_slot: 0-90 对应的时间槽 (7天*13小时)
        """
        if self.loads_curve is None:
            return
            
        # 将time_slot映射到具体的天和工作时间索引
        day_idx, hour_idx = self.time_slot_to_day_hour(time_slot)
        
        # 工作时间是6-18点，hour_idx是0-12的索引
        if 0 <= hour_idx <= 12:
            actual_hour = 6 + hour_idx  # 6-18点
        else:
            actual_hour = 12  # 默认使用12点
            print(f"[警告] 时间槽{time_slot}对应的工作时间索引{hour_idx}超出范围0-12")
        
        # 计算在168小时数据中的索引
        full_hour_idx = day_idx * 24 + actual_hour
        col_name = f'hour_{full_hour_idx}'
        
        if col_name not in self.loads_curve.columns:
            print(f"[警告] 列{col_name}不存在，使用默认负荷")
            return
        
        # 应用负荷到每个负荷节点
        for load_idx in self.net.load.index:
            if load_idx in self.loads_curve.index:
                load_value = self.loads_curve.loc[load_idx, col_name]
                self.net.load.at[load_idx, "p_mw"] = load_value
            else:
                # 如果某个负荷索引不在CSV中，使用基础值
                self.net.load.at[load_idx, "p_mw"] = 0.5
        
        print(f"[负荷应用] 时间槽{time_slot} -> 第{day_idx+1}天{actual_hour:02d}点 (hour_{full_hour_idx})")


    def is_radial(self):
        multi_graph = create_nxgraph(self.net)   # multi_graph是MultiGraph
        G = nx.Graph(multi_graph)                # 转为普通无重边无向图
        return len(nx.cycle_basis(G)) == 0
    
    def calc_total_reward(self):
        """计算综合reward"""
        total_v_deviation = 0.0
        total_important_dev = 0.0

        for t in range(self.time_slots):
            # 获取当前时间槽信息
            time_info = self.get_time_info(t)
            
            self.restore_net()
            self.apply_loads_curve(t)
            
            # 打印当前仿真时间（调试用）
            if t % 13 == 0:  # 每天开始时打印
                print(f"[仿真进度] {time_info['time_str']}")
            
            # 应用所有在t时刻活跃的检修与转供操作
            active_tasks = []
            for task in self.tasks:
                if task['status'] == 'assigned' and \
                task['assigned_time_idx'] <= t < task['assigned_time_idx'] + task['duration']:
                    active_tasks.append(task)
                    self.deactivate_element(task)
                    self.apply_transfer_scheme(task)
            
            # 如果有活跃任务，打印调试信息
            if active_tasks:
                task_names = [t['element_name'] for t in active_tasks]
                print(f"[{time_info['time_str']}] 活跃任务: {task_names}")
            
            # 潮流计算
            try:
                pp.runpp(self.net, numba=False)
            except Exception as e:
                print(f"[reward-debug] loadflow fails at {time_info['time_str']}: {e}")
                total_v_deviation += 100
                total_important_dev += 100
                continue  # ✅ 潮流失败，跳到下一个时间步
            
            # ✅ 潮流成功后才进行后续检查
            # 环网检查
            if not self.is_radial():
                print(f"[reward-debug] 非辐射结构 at {time_info['time_str']}，极大惩罚！")
                total_v_deviation += 100
                total_important_dev += 100
                continue  # ✅ 非辐射状，跳到下一个时间步
            
            # ✅ 网络正常，计算电压偏差
            # 全网电压偏差
            voltages = self.net.res_bus.vm_pu.values
            abs_dev = np.abs(voltages - 1.0)
            
            # 简化调试信息（避免输出过多）
            if active_tasks or abs_dev.max() > 0.1:  # 只在有任务或电压偏差大时打印
                print(f"[reward-debug] {time_info['time_str']}, mean dev={abs_dev.mean():.4f}, max dev={abs_dev.max():.4f}")
            
            total_v_deviation += abs_dev.sum()
            
            # 重要用户电压偏差
            if 'important' in self.net.bus.columns:
                important_dev_sum = 0.0
                for bus_idx, bus in self.net.bus.iterrows():
                    if bus['important']:
                        v = self.net.res_bus.at[bus_idx, "vm_pu"]
                        dev = abs(v - 1.0)
                        important_dev_sum += dev
                        
                        # 只在电压偏差较大时打印
                        if dev > 0.05:  # 超过5%才打印
                            print(f"[重要用户] {time_info['time_str']}, bus {bus_idx}, V={v:.4f}, dev={dev:.4f}")
                
                total_important_dev += important_dev_sum

        # 计算最终奖励
        reward = -total_v_deviation - 3.0 * total_important_dev
        print(f"[奖励计算] 总电压偏差={total_v_deviation:.4f}, 重要用户偏差={total_important_dev:.4f}, 最终奖励={reward:.4f}")
        
        return reward

#=======================观测函数组=============================
    def get_obs_agent(self, agent_id):
        task_features = self._get_task_features(agent_id)

        return np.concatenate([
            task_features, 
        ])

    def _get_task_features(self, agent_id):
        """任务相关特征"""
        next_task = self._get_next_unassigned_task(agent_id)
        
        if next_task is None:
            return np.zeros(15)  # 没有任务时的占位符
        
        # 任务基本信息
        task_features = [
            next_task['priority'] / 3.0,                           # 归一化优先级
            next_task['duration'] / 24.0,                          # 归一化持续时间
            len(next_task['allowed_time_slots']) / 91.0,           # 时间灵活性
            len(next_task['transfer_options']) / 10.0,             # 转供方案数量
            
            # 时间紧迫性
            min(next_task['allowed_time_slots']) / 91.0,           # 最早可执行时间
            max(next_task['allowed_time_slots']) / 91.0,           # 最晚可执行时间
            
        ]
        
        return np.array(task_features)
#======================================================================================

    def get_state(self):
        """全局状态向量"""
        feat = []
        total_tasks = len(self.tasks) if self.tasks else 1  # 避免除零
        
        for v in [1,2,3]:
            count = sum(1 for t in self.tasks if t["priority"]==v and t["status"]!="cancelled")
            feat.append(float(count)/float(total_tasks))  # ✅ 确保是Python float
        
        feat.append(0.0)  # ✅ 使用Python float
        
        return np.array(feat, dtype=np.float32)

    def get_obs(self):
        """所有agent的观测"""
        return [self.get_obs_agent(i) for i in range(self.agent_num)]
    

    def get_total_actions(self):
        """计算考虑优先级压缩后的最大动作空间"""
        max_transfer_options = 0
        for feeder_id in range(1, 5):
            transfer_schemes = get_feeder_transfer_schemes(feeder_id)
            max_transfer_options = max(max_transfer_options, len(transfer_schemes))
        
        return self.time_slots * max_transfer_options
    # 添加辅助函数：动作解码
    def decode_action(self, action_id):
        """
        将展平的动作ID解码为(时间槽, 转供方案索引)
        """
        max_transfer_options = self.get_total_actions() // self.time_slots
        time_slot = action_id // max_transfer_options
        transfer_idx = action_id % max_transfer_options
        return time_slot, transfer_idx

    def encode_action(self, time_slot, transfer_idx):
        """
        将(时间槽, 转供方案索引)编码为展平的动作ID
        """
        max_transfer_options = self.get_total_actions() // self.time_slots
        return time_slot * max_transfer_options + transfer_idx

    def get_env_info(self):
        return {
            "n_actions": int(self.get_total_actions()),  # 91 * max_transfer = 正确的展平动作空间
            "state_shape": int(self.get_state().shape[0]),
            "obs_shape": int(self.get_obs_agent(0).shape[0]),
            "n_agents": int(self.n_agents),
            "episode_limit": int(self.episode_limit),
        }
