import numpy as np
import pandas as pd
import pandapower as pp
import os

# ---- 固定随机种子和警告过滤 ----
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")

# ---- 基本参数 ----
DAYS = 7
HOURS_PER_DAY = 24
TOTAL_HOURS = DAYS * HOURS_PER_DAY  # 7*24=168小时

GRID = 'ober_sub'
OUTDIR = f'data/{GRID}'
os.makedirs(OUTDIR, exist_ok=True)

# ---- 1. 初始化网络 ----
net = pp.networks.mv_oberrhein(
    scenario='generation',
    cosphi_load=0.98,
    cosphi_pv=1.0,
    include_substations=False,
    separation_by_sub=False
)

print(f"Bus数量: {net.bus.shape[0]}   Line数量: {net.line.shape[0]}   Load数量: {net.load.shape[0]}")
print(f"负荷索引范围: {net.load.index.min()} - {net.load.index.max()}")
print(f"负荷索引列表: {net.load.index.tolist()}")

# ---- 2. 定义典型日负荷曲线模板 ----
def get_daily_load_pattern():
    """
    生成24小时标准化负荷模式 (0-23点)
    返回: list of 24个负荷系数
    """
    pattern = []
    for hour in range(24):
        if hour in [0, 1, 2, 3, 4, 5]:      # 深夜低谷 00-05
            factor = 0.3
        elif hour in [6, 7]:                # 早晨起升 06-07  
            factor = 0.5
        elif hour in [8, 9, 10]:            # 上午高峰 08-10
            factor = 0.85
        elif hour in [11]:                  # 上午回落 11
            factor = 0.7
        elif hour in [12, 13, 14]:          # 午间低谷 12-14
            factor = 0.45
        elif hour in [15, 16]:              # 下午回升 15-16
            factor = 0.75
        elif hour in [17, 18, 19]:          # 晚高峰 17-19
            factor = 0.9
        elif hour in [20, 21]:              # 夜间回落 20-21
            factor = 0.7
        elif hour in [22, 23]:              # 夜间低谷 22-23
            factor = 0.4
        else:
            factor = 0.6  # 默认
        
        # 添加小幅随机波动
        factor += np.random.normal(0, 0.03)
        factor = np.clip(factor, 0.1, 1.0)
        pattern.append(factor)
    
    return pattern

# ---- 3. 为每个负荷生成7天完整负荷曲线 ----
rows = []

for load_idx in net.load.index:  # 使用实际的负荷索引 (100-146)
    load_info = net.load.loc[load_idx]
    bus_id = load_info['bus']
    name = load_info['name']
    base_power = load_info['p_mw']
    
    # 负荷类型判断及系数
    if 'R' in name:
        load_type = 'residential'
        type_factor = 1.0
        weekend_factor = 0.8  # 居民负荷周末略低
    elif 'CI' in name:
        load_type = 'commercial'  
        type_factor = 1.2
        weekend_factor = 0.6  # 商业负荷周末显著降低
    elif 'LV' in name:
        load_type = 'residential_LV'
        type_factor = 0.9
        weekend_factor = 0.8
    elif 'MV' in name:
        load_type = 'industrial_MV'
        type_factor = 1.1
        weekend_factor = 0.7  # 工业负荷周末降低
    else:
        load_type = 'other'
        type_factor = 1.0
        weekend_factor = 0.8
    
    # 生成7天的负荷曲线
    load_curve = []
    
    for day in range(DAYS):
        # 判断是否为周末 (假设第6、7天为周末)
        is_weekend = day >= 5
        day_type_factor = weekend_factor if is_weekend else 1.0
        
        # 每天的随机变化因子
        daily_random_factor = np.random.uniform(0.85, 1.15)
        
        # 获取当天的24小时负荷模式
        daily_pattern = get_daily_load_pattern()
        
        for hour in range(24):
            # 计算该小时的负荷值
            load_value = (base_power * 
                         type_factor * 
                         day_type_factor * 
                         daily_random_factor * 
                         daily_pattern[hour])
            
            # 添加小幅随机波动
            load_value *= np.random.uniform(0.95, 1.05)
            
            # 确保最小负荷
            load_value = max(0.05 * base_power, load_value)
            
            load_curve.append(load_value)
    
    # 构建该负荷的数据行
    row_data = {'load_idx': load_idx}
    
    # 添加所有168小时的数据
    for hour_idx in range(TOTAL_HOURS):
        row_data[f'hour_{hour_idx}'] = load_curve[hour_idx]
    
    rows.append(row_data)

# ---- 4. 创建DataFrame ----
df_curve = pd.DataFrame(rows)
df_curve.set_index('load_idx', inplace=True)

# ---- 5. 输出到文件 ----
output_file = "test_loads_curve.csv"
df_curve.to_csv(output_file)
print(f"[OK] 完整7天负荷曲线已保存至: {output_file}")

# ---- 6. 验证和统计信息 ----
print(f"\n=== 负荷曲线生成完成 ===")
print(f"负荷数量: {len(df_curve)} (网络负荷数: {len(net.load)})")
print(f"时间列数: {len(df_curve.columns)} (7天*24小时 = {TOTAL_HOURS})")
print(f"索引范围: {df_curve.index.min()} - {df_curve.index.max()}")
print(f"索引匹配: {set(df_curve.index) == set(net.load.index)}")

# ---- 7. 显示各时段统计 ----
print(f"\n=== 典型日负荷统计 (第1天) ===")
for hour in range(24):
    col = f'hour_{hour}'
    if col in df_curve.columns:
        mean_val = df_curve[col].mean()
        std_val = df_curve[col].std()
        print(f"  {hour:02d}点: 平均{mean_val:.3f}MW, 标准差{std_val:.3f}MW")

# ---- 8. 显示工作时间段统计 ----
print(f"\n=== 工作时间段负荷统计 (6-18点) ===")
work_hours = list(range(6, 19))  # 6-18点
for day in range(DAYS):
    day_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][day]
    work_load_sum = 0
    for hour in work_hours:
        hour_idx = day * 24 + hour
        col = f'hour_{hour_idx}'
        work_load_sum += df_curve[col].mean()
    
    avg_work_load = work_load_sum / len(work_hours)
    print(f"  {day_name}: 工作时间平均负荷 {avg_work_load:.3f}MW")

# ---- 9. 生成时间索引对照表 ----
print(f"\n=== 时间索引对照表 (前24小时示例) ===")
for hour_idx in range(24):
    day = hour_idx // 24
    hour = hour_idx % 24
    day_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][day]
    print(f"  hour_{hour_idx:03d} -> {day_name} {hour:02d}:00")

print(f"\n=== 文件格式 ===")
print(f"CSV格式: load_idx, hour_0, hour_1, ..., hour_{TOTAL_HOURS-1}")
print(f"时间映射: hour_X 对应第(X//24+1)天的第(X%24)小时")
print(f"工作时间筛选: 在环境中选择每天的6-18点数据即可")