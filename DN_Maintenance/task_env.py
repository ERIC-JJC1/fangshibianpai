import numpy as np
from datetime import datetime, timedelta
from multiagentenv import MultiAgentEnv  # PyMARL基类
import random  
import pandas as pd   
import copy
import pandapower as pp  
import pandapower.networks as pn  
import pandapower.plotting as ppp  
import networkx as nx  
import copy  
import pandas as pd
from datetime import datetime
import itertools


# 设定馈线之间的联络开关
# 这里只是示意，实际你可以灵活组合——哪个馈线因哪条支路可转供到哪个邻馈线，哪些开关需要操作
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

#设定转供方案
def get_feeder_transfer_schemes(feeder_id):
    candidate_lines = inter_feeder_switch_map[feeder_id]  # 本馈线所有联络
    n = len(candidate_lines)
    transfer_list = []
    transfer_list.append({'id': 0, 'desc': '不转供', 'switch_ops': []})  # 基础动作
    idx = 1
    # 生成1~n条联络线的所有组合（排除全不选的组合）
    for k in range(1, n+1):
        for combo in itertools.combinations(candidate_lines, k):
            switch_ops = []
            desc_items = []
            # 把每条联络线上所有需要操作的开关都合闭
            for link in combo:
                desc_items.append(f"{link['with_feeder']} via line {link['line']}")
                # 默认两端都闭合
                switch_ops.append({'switch_id': link['switch_pair'][0], 'closed': True})
                switch_ops.append({'switch_id': link['switch_pair'][1], 'closed': True})
            transfer_list.append({
                'id': idx,
                'desc': '转供: ' + ', '.join(desc_items),
                'switch_ops': switch_ops
            })
            idx += 1
    return transfer_list


#----------------------读取excel文件--------------------------------
def read_tasks_from_excel(file_path):
    df = pd.read_excel(file_path)
    tasks = []
    for idx, row in df.iterrows():
        # 跳过无效行
        if pd.isnull(row['所属馈线']) or pd.isnull(row['设备名称']) or pd.isnull(row['开始时间']) or pd.isnull(row['持续时间(小时)']) or pd.isnull(row['优先级']):
            continue
        
        # 读取对应信息
        feeder_id = int(row['所属馈线'])          # 馈线编号，需与环境的馈线ID对齐
        element_name = str(row['设备名称'])       # 元件名称
        start_time = pd.to_datetime(row['开始时间'])
        duration = int(row['持续时间(小时)'])
        priority = int(row['优先级'])           # 优先级：1低，2中，3高
        
        # 元件类型自动判别（假设名称开头如“bus123”）
        if element_name.startswith('bus'):
            element_type = 'bus'
            element_id = int(element_name[4:])
        elif element_name.startswith('line'):
            element_type = 'line'
            element_id = int(element_name[5:])
        elif element_name.startswith('switch'):
            element_type = 'switch'
            element_id = int(element_name[7:])
        elif element_name.startswith('trafo'):
            element_type = 'trafo'
            element_id = int(element_name[6:])
        else:
            element_type = 'unknown'
            element_id = None
        
        # 构造任务字典
        task = {
            'feeder_id': feeder_id,  # 馈线ID
            'element_type': element_type,
            'element_id': element_id,
            'element_name': element_name,
            'start_time': start_time.strftime("%Y-%m-%d %H:%M"),
            'duration': duration,
            'priority': priority,
            'status': 'unassigned',  # 初始未分配
            # 可以补充如'nickname'、'needs_transfer'等
        }
        tasks.append(task)
    return tasks

# 使用
file_path = 'maintenance_plan_20250521_151713.xlsx'
tasks = read_tasks_from_excel(file_path)
print("已读取任务数：", len(tasks))
for t in tasks[:3]:
    print(t)   # 打印前3个任务字典
    break

# 假设tasks是所有已读取任务
for task in tasks:
    feeder_id = task['feeder_id']
    transfer_options = get_feeder_transfer_schemes(feeder_id)
    task['transfer_options'] = transfer_options
    # 可加mask，预留后续支持更复杂限制
    task['transfer_mask'] = np.ones(len(transfer_options), dtype=np.int_)

#--------------------------------------------------------------------------

##--------------------------定义转供方案和实际操作开关映射的函数---------------------------------------

def apply_transfer_scheme(self, task):
    transfer_list = task['transfer_options']
    transfer_idx = task.get('transfer_idx', 0)
    switch_ops = transfer_list[transfer_idx]['switch_ops']
    for op in switch_ops:
        self.net.switch.loc[op['switch_id'], "closed"] = op['closed']

 #----------------------------------------------------------------------------------------

 
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

 #---------------------------------------------------------------

# 馈线参数根据你的现场网络实际调整  

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
    region_assets = {
        i: {
            "buses": region_buses[i],
            "lines": region_lines[i],
            "switches": region_switches[i],
            "trafos": region_trafos[i],
        } for i in range(4)
    }
    return region_assets


class GridMaintenanceEnv(object):
        

    def __init__(self, net, initial_tasks, forecast_loads, horizon_days=7, window_size=3):

        super().__init__()
        self.net = net
        important_buses = [167, 273, 244, 65, 148, 216, 227]  
        net.bus["important"] = False  
        net.bus.loc[important_buses, "important"] = True
        G = nx.Graph(net)
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
        feeders_bus_sets = [set(feeder1), set(feeder2), set(feeder3), set(feeder4)]

        self.region_assets = build_region_assets(net, feeders_bus_sets)
        self.agent_num =4# 明确智能体数量
        self.agents = list(range(self.agent_num))
        self.tasks_all = [dict(t) for t in initial_tasks]
        self.forecast_loads = forecast_loads
        self.horizon = horizon_days
        self.window_size = window_size
        self.day_num = 7
        self.hours = list(range(6, 19))  # 6-18点，共13小时
        self.time_slots = self.day_num * len(self.hours)  # 91

        for task in initial_tasks:
            # 先判断bus，再看line再看switch/trafo（如果有多个，只判第一个bus或主设备，可根据实际需求自定义）
            assigned = False
            for agent_id in range(self.agent_num):
                if 'affected_buses' in task and set(task['affected_buses']) & set(self.region_assets[agent_id]['buses']):
                    task['region_id'] = agent_id
                    assigned = True
                    break
                elif 'affected_lines' in task and set(task['affected_lines']) & set(self.region_assets[agent_id]['lines']):
                    task['region_id'] = agent_id
                    assigned = True
                    break
                elif 'affected_switches' in task and set(task['affected_switches']) & set(self.region_assets[agent_id]['switches']):
                    task['region_id'] = agent_id
                    assigned = True
                    break
                elif 'affected_trafos' in task and set(task['affected_trafos']) & set(self.region_assets[agent_id]['trafos']):
                    task['region_id'] = agent_id
                    assigned = True
                    break
            if not assigned:
                task['region_id'] = -1  # 标记为跨区，后续专门处理

        self.episode_limit = 24 * self.horizon  # 多少步结束

        self.reset()


    def reset(self):
        # 还原全部任务与时间表
        self.current_time = datetime.strptime(min([t["start_time"] for t in self.tasks_all]), "%Y-%m-%d %H:%M")
        self.tasks = [dict(t) for t in self.tasks_all]
        self.steps = 0
        # 可选：记录每个agent上回观测
        return self.get_obs(), self.get_state()


    def step(self, actions):
        # actions: [actions_agent0, actions_agent1, ...]
        # 1. 任务分配

        for agent_id in range(self.agent_num):
            region_tasks = [t for t in self.tasks if t['region_id'] == agent_id]
            for i, (start, duration, transfer) in enumerate(actions[agent_id]):
                # 限制transfer不越界选项
                transfer_options = region_tasks[i]['transfer_options']
                transfer = int(transfer)
                if transfer >= len(transfer_options):
                    transfer = 0  # 或raise
                region_tasks[i]['assigned_time_idx'] = start
                region_tasks[i]['duration'] = duration
                region_tasks[i]['transfer_idx'] = transfer
                region_tasks[i]['status'] = 'assigned'

        # 2. 仿真与reward
        total_reward = self.calc_total_reward()
        obs = self.get_obs()
        state = self.get_state()
        done = all(task['status'] == 'assigned' for task in self.tasks)
        info = {}
        return obs, total_reward, done, info


    #动作的奖励函数计算#
    def calc_total_reward(self):
        """
        计算一个episode内所有任务分配方案的综合reward——
        汇总所有时段下全网电压偏差与重要用户电压偏差。
        """
        total_v_deviation = 0.0
        total_important_dev = 0.0

        for t in range(self.time_slots):
            self.restore_net()
            # 应用所有在t时刻活跃的检修与转供操作
            for task in self.tasks:
                if task['status'] == 'assigned' and \
                task['assigned_time_idx'] <= t < task['assigned_time_idx'] + task['duration']:
                    self.deactivate_element(task)
                    self.apply_transfer_scheme(task)
            try:
                pp.runpp(self.net, numba=False)
            except Exception as e:
                # 潮流不收敛直接重罚
                total_v_deviation += 9999
                total_important_dev += 9999
                continue

            # 全网电压偏差（所有节点|V-1.0|之和）
            voltages = self.net.res_bus.vm_pu.values
            total_v_deviation += np.abs(voltages - 1.0).sum()

            # 重要用户电压偏差（全部important bus的|V-1.0|之和）
            # 如果你的重要用户字段不是important，而是其他名称，告诉我改这里即可
            if 'important' in self.net.bus:
                for bus_idx, bus in self.net.bus.iterrows():
                    if bus['important']:
                        v = self.net.res_bus.at[bus_idx, "vm_pu"]
                        total_important_dev += abs(v - 1.0)
            # 如果没有important字段，请补充或告诉我用别的字段

        # 你可自由调整权重
        reward = -total_v_deviation - 3.0 * total_important_dev
        return reward


    def get_obs_agent(self, agent_id):
        """返回每个智能体自己的观测向量"""
        region = self.agents[agent_id]
        tasks_window = [t for t in self.tasks if t.get("region_id")==region and t["start_time"] >= self.current_time.strftime("%Y-%m-%d %H:%M") and t["status"]!="cancelled"]
        tasks_window = sorted(tasks_window, key=lambda x: x["start_time"])
        tasks_window = tasks_window[:self.window_size]
        features = []
        for t in tasks_window:
            features.extend([
                t["priority"]/3.0,
                1.0 if t["task_type"]=="node" else 0.0,
                1.0 if t.get("needs_transfer", False) else 0.0
            ])
        while len(features) < self.window_size*3:
            features.append(0)
        # 可加入区域负荷预测、重要节点指标
        return np.array(features, dtype=np.float32)


    def get_obs(self):
        """所有agent的观测"""
        return [self.get_obs_agent(i) for i in range(self.agent_num)]


    def get_state(self):
        """全局状态向量，供集中训练时用（如所有任务某些状态，系统总指标等）"""
        # 举例：全部未完成任务的优先级比例+最大电压风险等
        feat = []
        for v in [1,2,3]:
            feat.append(sum(1 for t in self.tasks if t["priority"]==v and t["status"]!="cancelled")/len(self.tasks))
        feat.append(0)  # 可加入全网最大|V-1|之类
        return np.array(feat, dtype=np.float32)




#------------------------获得被动作的开关------------------------
    def get_occupied_switches(self):
        """获得当前所有已被闭合的联络开关（如[(106,107), ...]）"""
        occupied = set()
        for task in self.tasks:
            if task['status'] == 'assigned':
                transfer_idx = task.get('transfer_idx', 0)
                # 哪些开关被已分配任务“闭合”
                ops = task['transfer_options'][transfer_idx]['switch_ops']
                for op in ops:
                    if op['closed']:
                        occupied.add(op['switch_id'])
        return occupied   
    



    def get_avail_agent_actions(self, agent_id):
        """
        返回指定agent当前所有未分配任务的动作空间掩码
        动作空间含义：
        - start_time_idx（分时掩码mask，长度为可选起始时间槽个数）
        - duration_idx  （此场景恒定，只返回1，给RL用作动作占位）
        - transfer_idx  （转供动作空间，对应转供方案数量）
        """
        # ------- 1. 获取未分配的任务 -------
        region_tasks = [
            t for t in self.tasks
            if t.get('region_id') == agent_id and t['status'] == 'unassigned'
        ]
        occupied_switches = self.get_occupied_switches()
        avail_list = []
        for task in region_tasks:
            # ------- 2. 时间掩码 -------
            # 持续时间，单位：小时
            duration = int(task['duration'])
            # 可选时间槽（一天13个：6~18，总共7天=91）
            valid_time_mask = [0]*self.time_slots  # self.time_slots=91

            # 是否特殊检修类型（如需夜间、高峰或连续作业等，将其属性如'time_limit':'free'/'special')
            time_limit_type = task.get('time_limit', 'normal')

            for idx in range(self.time_slots):
                # 计算该动作对应的（第几天、第几小时）
                day_idx = idx // 13
                hour = 6 + (idx % 13)   # hour取6~18

                end_hour = hour + duration
                # 注意这里严格按你需求，结束必须 <=22点
                if time_limit_type == 'special':
                    # 可以允许更灵活，如00点~23点等（留接口）
                    valid_time_mask[idx] = 1
                else:
                    if hour >= 6 and end_hour <= 22:
                        valid_time_mask[idx] = 1
            # ------- 3. 持续时长掩码 -------
            # 固定duration，不可选，只给1
            duration_mask = [1]

        new_transfer_mask = []
        for option in task['transfer_options']:
            # 检查该option下所有switch_ops是否都不与occupied_switches冲突
            can_choose = True
            for op in option['switch_ops']:
                if op['closed'] and op['switch_id'] in occupied_switches:
                    can_choose = False
                    break
            new_transfer_mask.append(1 if can_choose else 0)
        avail_list.append((valid_time_mask, duration_mask, new_transfer_mask))
        
        return avail_list
    

    def get_total_actions(self):
        return self.time_slots  # 91


    def get_env_info(self):
        return {
            "state_shape": self.get_state().shape[0],
            "obs_shape": self.get_obs_agent(0).shape[0],
            "n_actions": 4,
            "n_agents": self.agent_num,
            "episode_limit": self.episode_limit
        }
