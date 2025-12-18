import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import copy

# 引入你的新拓扑定义
from topology.tie_map import FEEDER_BUS_SETS, TIE_SWITCHES

class MaintenanceTaskGenerator:
    def __init__(self, net):
        self.net = net
        self.feeder_map = self._build_bus_to_feeder_map()
        
    def _build_bus_to_feeder_map(self):
        """将集合映射反转为 节点->馈线ID 的字典，方便查询"""
        bus_map = {}
        for fid, buses in FEEDER_BUS_SETS.items():
            for bus in buses:
                bus_map[bus] = fid
        return bus_map

    def get_feeder_id(self, bus_id):
        return self.feeder_map.get(bus_id, -1)

    def find_transfer_options(self, feeder_id):
        """
        根据新的 Tie Map，查找当前馈线有哪些转供"外援"
        返回: [{'switch_id': 106, 'target_feeder': 2}, ...]
        """
        options = []
        for sw_id, (f1, f2) in TIE_SWITCHES.items():
            if f1 == feeder_id:
                options.append({'switch_id': sw_id, 'target_feeder': f2})
            elif f2 == feeder_id:
                options.append({'switch_id': sw_id, 'target_feeder': f1})
        return options

    def generate_tasks(self, num_tasks=5, start_date="2025-06-01", days_span=7):
        """生成标准化检修任务列表"""
        tasks = []
        
        # 1. 筛选合法的检修对象 (在线路表中筛选)
        # 排除掉那些一断开就全黑的末端线路（可选，为了增加难度也可以保留）
        candidate_lines = self.net.line.index.tolist()
        
        # 随机采样
        selected_lines = random.sample(candidate_lines, min(num_tasks, len(candidate_lines)))
        
        for i, line_id in enumerate(selected_lines):
            # 获取线路两端节点
            f_bus = self.net.line.at[line_id, 'from_bus']
            t_bus = self.net.line.at[line_id, 'to_bus']
            
            # 确定所属馈线 (取两端任意一个属于已知馈线的)
            feeder_id = self.get_feeder_id(f_bus)
            if feeder_id == -1: feeder_id = self.get_feeder_id(t_bus)
            
            # 查找该馈线的转供方案
            transfer_opts = self.find_transfer_options(feeder_id)
            
            # 生成时间
            day_offset = random.randint(0, days_span - 1)
            hour_start = random.randint(8, 14) # 检修通常在白天
            duration = random.choice([2, 4, 6])
            
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day_offset, hours=hour_start)
            
            task = {
                "task_id": f"T{i+1:03d}",
                "target_element": "line",
                "element_id": line_id,
                "feeder_id": feeder_id,
                "start_time": start_dt.strftime("%Y-%m-%d %H:00"),
                "duration": duration,
                "transfer_candidates": transfer_opts, # 这里直接带上了可用的联络开关
                "description": f"Line {line_id} Maintenance (Feeder {feeder_id})"
            }
            tasks.append(task)
            
        return tasks

    def generate_realistic_tasks(self, n_nodes=5, n_single_lines=5, n_single_switches=5, n_single_trafos=1,
                                 start_date="2025-06-01", days_span=7, seed=None):
        """
        生成真实的检修计划，使用从DN_Maintenance移植的高级功能
        """
        # 导入高级任务生成功能
        from task.advanced_task_generator import generate_realistic_maintenance_plan
        
        # 调用高级任务生成函数
        tasks = generate_realistic_maintenance_plan(
            self.net,
            FEEDER_BUS_SETS,
            self.feeder_map,
            n_nodes=n_nodes,
            n_single_lines=n_single_lines,
            n_single_switches=n_single_switches,
            n_single_trafos=n_single_trafos,
            start_date=start_date,
            days_span=days_span,
            seed=seed
        )
        
        return tasks

if __name__ == "__main__":
    # 测试代码
    import toy_network_jjc_179 as tn
    net = tn.net
    
    gen = MaintenanceTaskGenerator(net)
    
    # 生成简单任务
    tasks = gen.generate_tasks(3)
    print("=== 生成任务预览 ===")
    for t in tasks:
        print(t)
    
    # 生成高级任务
    print("\n=== 生成高级任务预览 ===")
    advanced_tasks = gen.generate_realistic_tasks(n_nodes=2, n_single_lines=2, n_single_switches=2)
    for t in advanced_tasks[:3]:  # 只显示前3个任务
        print(f"任务ID: {t['task_id']}, 类型: {t['task_type']}, 描述: {t['desc']}")