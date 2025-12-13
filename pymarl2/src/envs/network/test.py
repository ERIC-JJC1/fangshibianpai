import pandapower.networks as pn
import numpy as np
import random
from powergrid_env import read_tasks_from_excel, GridMaintenanceEnv

def comprehensive_test():
    """综合测试PowerGrid环境"""
    
    print("="*60)
    print("          配电网检修调度环境综合测试")
    print("="*60)
    
    # ============ 1. 环境初始化测试 ============
    print("\n[测试1] 环境初始化...")
    try:
        tasks = read_tasks_from_excel('maintenance_plan_20250521_151713.xlsx')
        print(f"✅ 成功读取 {len(tasks)} 个任务")
        
        env = GridMaintenanceEnv(
            initial_tasks=tasks, 
            forecast_loads=None,
            load_curve_file='pymarl2/src/envs/network/test_loads_curve.csv'
        )
        print("✅ 环境初始化成功")
        
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return
    
    # ============ 2. 环境信息测试 ============
    print("\n[测试2] 环境信息...")
    env_info = env.get_env_info()
    print(f"✅ 智能体数量: {env_info['n_agents']}")
    print(f"✅ 状态空间: {env_info['state_shape']}")
    print(f"✅ 观测空间: {env_info['obs_shape']}")

    
    # ============ 3. 重置测试 ============
    print("\n[测试3] 环境重置...")
    obs, state,avail_actions = env.reset()
    print(f"✅ 观测形状: {[o.shape for o in obs]}")
    print(f"✅ 状态形状: {state.shape}")
    
    # ============ 4. 时间编码测试 ============
    print("\n[测试4] 时间编码验证...")
    test_time_slots = [0, 1, 12, 13, 25, 90]
    for slot in test_time_slots:
        if slot < env.time_slots:
            time_info = env.get_time_info(slot)
            print(f"  时间槽{slot:2d} → {time_info['time_str']}")
    
    
    # ============ 6. 合理动作生成与执行 ============
    print("\n[测试6] 生成合理动作...")
    actions = generate_valid_actions(env)
    print("✅ 动作生成完成")
    
    # 打印动作详情
    for agent_id, agent_actions in enumerate(actions):
        print(f"  Agent {agent_id}: {len(agent_actions)} 个动作")
        for i, (start_time, transfer_idx) in enumerate(agent_actions):
            print(f"    动作{i}: 时间槽{start_time}, 转供方案{transfer_idx}")
    
    # ============ 7. 执行动作测试 ============
    print("\n[测试7] 执行动作...")
    try:
        obs, reward, done, info = env.step(actions)
        print(f"✅ 动作执行成功")
        print(f"✅ 奖励: {reward:.4f}")
        print(f"✅ 完成状态: {done}")
        
    except Exception as e:
        print(f"❌ 动作执行失败: {e}")
        return
    
    # ============ 8. 转供状态验证 ============
    print("\n[测试8] 转供状态验证...")
    verify_transfer_schemes(env)
    
    # ============ 9. 边界条件测试 ============
    print("\n[测试9] 边界条件测试...")
    test_boundary_conditions(env)
    
    # ============ 10. 多轮测试 ============
    print("\n[测试10] 多轮执行测试...")
    test_multiple_episodes(env, num_episodes=3)
    
    print("\n" + "="*60)
    print("          🎉 所有测试完成！")
    print("="*60)


def generate_valid_actions(env):
    """生成合理的动作"""
    actions = []
    
    for agent_id in range(env.agent_num):
        region_tasks = [t for t in env.tasks if t['region_id'] == agent_id and t['status'] == 'unassigned']
        avail_actions = env.get_avail_actions()
        
        agent_actions = []
        for i, (time_mask, transfer_mask) in enumerate(avail_actions):
            # 选择一个有效的时间槽
            valid_times = [idx for idx, mask in enumerate(time_mask) if mask == 1]
            if valid_times:
                start_time = random.choice(valid_times)
            else:
                start_time = 0  # 默认
                
            # 选择一个有效的转供方案
            valid_transfers = [idx for idx, mask in enumerate(transfer_mask) if mask == 1]
            if valid_transfers:
                transfer_idx = random.choice(valid_transfers)
            else:
                transfer_idx = 0  # 默认不转供
                
            agent_actions.append((start_time, transfer_idx))
        
        actions.append(agent_actions)
    
    return actions

def verify_transfer_schemes(env):
    """验证转供方案执行情况"""
    mismatch_count = 0
    
    for task in env.tasks:
        if task.get('status') == 'assigned':
            transfer_idx = task.get('transfer_idx', 0)
            transfer_scheme = task['transfer_options'][transfer_idx]
            
            print(f"\n[任务] {task['element_name']}")
            print(f"  方案: {transfer_scheme['desc']}")
            
            for op in transfer_scheme['switch_ops']:
                expected = op['closed']
                actual = env.net.switch.at[op['switch_id'], 'closed']
                status = "✅" if expected == actual else "❌"
                
                print(f"  {status} Switch {op['switch_id']}: 期望{'合' if expected else '分'} → 实际{'合' if actual else '分'}")
                
                if expected != actual:
                    mismatch_count += 1
    
    if mismatch_count == 0:
        print("✅ 所有转供开关状态正确")
    else:
        print(f"❌ 发现 {mismatch_count} 个开关状态不符")

def test_boundary_conditions(env):
    """测试边界条件"""
    print("  测试非法动作...")
    
    # 测试越界动作
    illegal_actions = [
        [(999, 0)],  # 时间越界
        [(0, 999)],  # 转供方案越界
    ]
    
    for i, action_set in enumerate([[action] for action in illegal_actions]):
        try:
            # 构造完整动作（所有智能体）
            full_actions = [[] for _ in range(env.agent_num)]
            if action_set:
                full_actions[0] = action_set
            
            env.reset()
            obs, reward, done, info = env.step(full_actions)
            print(f"    非法动作{i+1}: 处理正常 (reward={reward:.2f})")
        except Exception as e:
            print(f"    非法动作{i+1}: 抛出异常 {e}")

def test_multiple_episodes(env, num_episodes=3):
    """测试多轮执行"""
    for episode in range(num_episodes):
        print(f"  Episode {episode+1}...")
        try:
            env.reset()
            actions = generate_valid_actions(env)
            obs, reward, done, info = env.step(actions)
            print(f"    ✅ Episode {episode+1} 完成, reward={reward:.4f}")
        except Exception as e:
            print(f"    ❌ Episode {episode+1} 失败: {e}")

if __name__ == "__main__":
    comprehensive_test()