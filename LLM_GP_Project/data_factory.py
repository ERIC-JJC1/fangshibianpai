# LLM_GP_Project/data_factory_v3.py
import pandas as pd
import pandapower as pp
import pandapower.topology as ppt
import numpy as np
from tqdm import tqdm
import random
import warnings
import toy_network_jjc_179 as tn
from task_generator import MaintenanceTaskGenerator

# 抑制 Pandapower 的冗余输出
warnings.filterwarnings("ignore")

def get_element_physics_info(net, element_type, element_id):
    """
    获取元件的物理画像，增强 Prompt 的信息量
    """
    info = ""
    if element_type == 'line':
        # 获取线路长度和阻抗信息
        length = net.line.at[element_id, 'length_km']
        r = net.line.at[element_id, 'r_ohm_per_km'] * length
        # 尝试估算下游负载（简单版：只看是否是主干）
        # 这里为了速度，暂不跑全网搜索，只记录静态参数
        info = f"Len:{length:.1f}km,R:{r:.2f}"
    return info

def run_data_factory(num_samples=3000, output_file="dataset_multitask_v1.csv"):
    # 1. 初始化网络
    net = tn.net 
    task_gen = MaintenanceTaskGenerator(net)
    
    dataset = []
    print(f"🏭 [V3工厂启动] 准备生成 {num_samples} 条【多任务并发】样本...")
    print("   -> 包含物理属性注入")
    print("   -> 包含 1-3 重故障组合")
    
    # 统计计数器
    stats = {"1_task": 0, "2_tasks": 0, "3_tasks": 0, "fail": 0}
    
    for i in tqdm(range(num_samples)):
        # --- A. 场景注入 (Scenario) ---
        # 扩大波动范围，覆盖极端工况
        load_scale = np.random.uniform(0.5, 1.4) 
        pv_scale = np.random.uniform(0.0, 1.0)
        
        # 备份原始状态
        original_load_p = net.load.p_mw.copy()
        original_sgen_p = net.sgen.p_mw.copy()
        original_line_status = net.line.in_service.copy()
        original_switch_status = net.switch.closed.copy()
        
        # 注入波动
        net.load.p_mw = original_load_p * load_scale
        net.sgen.p_mw = original_sgen_p * pv_scale
        
        # --- B. 多任务生成 (Multi-task Injection) ---
        # 概率分布：50%单任务，40%双任务，10%三任务
        n_tasks = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
        stats[f"{n_tasks}_tasks"] += 1
        
        tasks = task_gen.generate_tasks(n_tasks)
        
        # 构建 Prompt 组件
        task_prompts = []
        action_prompts = []
        
        # 执行任务（断线）
        for task in tasks:
            line_id = task['element_id']
            net.line.at[line_id, 'in_service'] = False
            
            # 获取物理信息增强 Prompt
            phy_info = get_element_physics_info(net, 'line', line_id)
            task_prompts.append(f"T{task['task_id']}:Line_{line_id}({phy_info},Fdr{task['feeder_id']})")
            
            # 随机决策转供 (Action Strategy)
            candidates = task['transfer_candidates']
            # 简单策略：如果有联络开关，50%概率闭合其中一个
            if candidates and random.random() > 0.5:
                choice = random.choice(candidates)
                sw_id = choice['switch_id']
                # 检查开关是否已经被之前的任务操作过，避免冲突
                if not net.switch.at[sw_id, 'closed']:
                    net.switch.at[sw_id, 'closed'] = True
                    action_prompts.append(f"Act{task['task_id']}:Close_Tie_{sw_id}")
                else:
                    action_prompts.append(f"Act{task['task_id']}:Shared_Tie_{sw_id}")
            else:
                action_prompts.append(f"Act{task['task_id']}:Islanding")
        
        # --- C. 构造最终 Prompt ---
        # 格式：[场景] || [任务列表] || [动作列表]
        scenario_str = f"Grid:Load={load_scale:.2f}x,PV={pv_scale:.2f}x"
        tasks_str = " & ".join(task_prompts)
        actions_str = " & ".join(action_prompts)
        
        full_prompt = f"{scenario_str} || {tasks_str} || {actions_str}"
        
        # --- D. 算真值 (Physics Simulation) ---
        min_vm = 0.0
        max_load = 999.0
        converged = 0
        
        try:
            # 检查是否有孤岛（Dead Island），如果有，pandapower可能会报错或算错
            # 这一步对于多任务非常重要！
            if ppt.unsupplied_buses(net):
                # 存在失电节点，直接判为严重违规，不跑潮流了（或者标记为特定值）
                converged = 0
                min_vm = 0.0 # 极刑
            else:
                pp.runpp(net)
                min_vm = net.res_bus.vm_pu.min()
                max_load = net.res_line.loading_percent.max()
                converged = 1
        except:
            converged = 0
            stats["fail"] += 1
            
        dataset.append({
            "prompt": full_prompt,
            "min_voltage": min_vm,
            "max_loading": max_load,
            "converged": converged,
            "num_tasks": n_tasks
        })
        
        # --- E. 严格复原 (Reset) ---
        net.line.in_service = original_line_status
        net.switch.closed = original_switch_status
        net.load.p_mw = original_load_p
        net.sgen.p_mw = original_sgen_p

    # 保存
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 完成！数据已保存至 {output_file}")
    print(f"📊 统计：{stats}")
    print("\n🔍 样本预览：")
    for p in df['prompt'].head(3):
        print(f"- {p}")

if __name__ == "__main__":
    run_data_factory()