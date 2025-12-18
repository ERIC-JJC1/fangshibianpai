import sys
import os
import numpy as np
import pandas as pd
import pandapower.networks as pn

# 添加项目路径
sys.path.append('/home/wzseu/JJC_project/fangshibianpai/DN_Maintenance')

from task_env import GridMaintenanceEnv, read_tasks_from_excel
from task_code import generate_realistic_maintenance_plan

def main():
    # 加载网络
    net = pn.mv_oberrhein()
    
    # 生成检修计划
    print("生成检修计划...")
    maintenance_plan = generate_realistic_maintenance_plan(
        net,
        n_nodes=5,
        n_single_lines=5,
        n_single_switches=5,
        n_single_trafos=2,
        start_date="2025-06-01",
        days_span=7,
        seed=42
    )
    
    # 创建简单的负荷预测数据（实际应用中应使用更精确的数据）
    forecast_loads = np.random.rand(7, 13)  # 7天，每天13小时（6-18点）
    
    # 初始化环境
    env = GridMaintenanceEnv(net, maintenance_plan, forecast_loads)
    
    # 获取环境信息
    env_info = env.get_env_info()
    print(f"环境信息: {env_info}")
    
    # 运行一个简单的随机策略来测试环境
    obs, state = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < env.episode_limit:
        # 随机动作（实际训练中会被强化学习算法替换）
        actions = []
        for agent_id in range(env.n_agents):
            agent_actions = []
            # 简化的动作选择（实际应该根据策略选择动作）
            start_time = np.random.randint(0, env.time_slots)
            duration = np.random.randint(1, 5)
            transfer = np.random.randint(0, 3)
            agent_actions.append((start_time, duration, transfer))
            actions.append(agent_actions)
        
        obs, reward, done, info = env.step(actions)
        episode_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"步骤 {steps}, 奖励: {reward}")
    
    print(f"回合结束，总奖励: {episode_reward}")

if __name__ == "__main__":
    main()