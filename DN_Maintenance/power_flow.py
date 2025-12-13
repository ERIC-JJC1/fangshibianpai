import numpy as np  
import pandas as pd  
import pandapower as pp  
import pandapower.plotting.plotly as pplotly  
import time  
import pickle  
from datetime import datetime, timedelta  
import random  
from tqdm import tqdm  
import os  
import copy 

# 导入检修计划生成模块  
from task_code import generate_realistic_maintenance_plan  

# 设置常量参数  
DAYS = 30                   # 模拟天数  
HOURS_PER_DAY = 24          # 每天小时数  
TIME_STEPS = DAYS * HOURS_PER_DAY  # 总时间步长  
P_NOISE = 0.02              # 功率测量噪声系数  
V_NOISE = 0.01              # 电压传感器噪声系数  
I_NOISE = 0.01              # 电流噪声系数  
POWER_COEF = 0.9            # 功率系数  
LOAD_DIST = 'normal'        # 负荷分布类型  

# 创建数据目录  
if not os.path.exists('data'):  
    os.makedirs('data')  

# 创建MV Oberrhein网络  
print("创建MV Oberrhein网络...")  
grid_name = 'ober_sub'  
net, _ = pp.networks.mv_oberrhein(scenario='generation', cosphi_load=0.98, cosphi_pv=1.0, include_substations=False, separation_by_sub=True)  

# 创建网络特定数据目录  
if not os.path.exists(f'data/{grid_name}'):  
    os.makedirs(f'data/{grid_name}')  

# 创建坐标用于可视化  
if 'geo' in net.bus.columns:  
    net.bus['geo'] = None  
if 'geo' in net.line.columns:  
    net.line['geo'] = None  
pp.plotting.create_generic_coordinates(net, respect_switches=True)  

# 创建日期范围  
start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  
date_range = [start_date + timedelta(hours=h) for h in range(TIME_STEPS)]  
date_str_range = [d.strftime("%Y-%m-%d %H:%M") for d in date_range]  

print(f"生成{TIME_STEPS}个时间点的负荷和发电曲线...")  

# 定义负荷曲线  
household_load_profile = np.array([0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.4, 0.65, 0.65, 0.65,   
                                  0.7, 0.6, 0.7, 0.65, 0.55, 0.5, 0.45, 0.6, 0.8, 0.9,   
                                  0.8, 0.7, 0.55, 0.3])  
industry_load_profile = np.array([0.35, 0.35, 0.3, 0.3, 0.4, 0.5, 0.6, 0.9, 1., 1.,   
                                 1., 0.9, 0.85, 0.85, 0.85, 0.85, 0.8, 0.55, 0.5, 0.45,   
                                 0.4, 0.4, 0.35, 0.35])  

# 定义发电曲线  
profile_day_sun = np.array([0., 0., 0., 0., 0., 0., 0.1, 0.25, 0.4, 0.7, 0.9, 1.,   
                           1., 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.1, 0., 0.])  
profile_day_wind = np.array([0.6, 0.6, 0.7, 0.5, 0.4, 0.4, 0.5, 0.7, 0.8, 0.7, 0.5, 0.5,   
                            0.4, 0.5, 0.4, 0.5, 0.6, 0.6, 0.3, 0.4, 0.7, 0.6, 0.4, 0.5])  

# 添加周末负荷变化  
is_weekend = np.array([date.weekday() >= 5 for date in date_range])  # 周六和周日为True  
weekend_factor = np.ones(TIME_STEPS)  
weekend_factor[is_weekend] = 0.8  # 周末负荷降低到工作日的80%  

# 添加季节性变化  
month_factors = {  
    1: 1.2,   # 一月 - 冬季用电高峰  
    2: 1.15,  # 二月  
    3: 1.0,   # 三月  
    4: 0.9,   # 四月  
    5: 0.85,  # 五月  
    6: 0.95,  # 六月 - 夏季开始  
    7: 1.1,   # 七月 - 夏季用电  
    8: 1.15,  # 八月 - 夏季高峰  
    9: 0.95,  # 九月  
    10: 0.9,  # 十月  
    11: 0.95, # 十一月  
    12: 1.1   # 十二月 - 冬季用电  
}  

season_factor = np.array([month_factors[date.month] for date in date_range])  

# 生成检修计划  
print("生成检修计划...")  
maintenance_start = start_date.strftime("%Y-%m-%d")  
maintenance_tasks = generate_realistic_maintenance_plan(  
    net,  
    n_nodes=5,                              # 节点级任务数量  
    n_single_lines=8,                       # 单独线路任务数量  
    n_single_switches=10,                   # 单独开关任务数量  
    n_single_trafos=3,                      # 单独变压器任务数量  
    start_date=maintenance_start,  
    days_span=DAYS,  
    seed=42  
)  

# 将检修任务转换为时间序列表示  
maintenance_schedule = {}  # {时间步: [{受影响设备类型: 设备ID}]}  

for task in maintenance_tasks:  
    # 解析任务开始时间  
    task_start = datetime.strptime(task["start_time"], "%Y-%m-%d %H:%M")  
    if task_start < start_date:  
        task_start = start_date  # 确保不会有过去的时间  
    
    task_duration = task["duration_h"]  
    
    # 找出任务时间范围内的所有时间步  
    for i, dt in enumerate(date_range):  
        if task_start <= dt < (task_start + timedelta(hours=task_duration)):  
            if i not in maintenance_schedule:  
                maintenance_schedule[i] = []  
            
            # 添加受影响设备信息  
            affected_equipment = {}  
            
            if task["task_type"] == "node":  
                # 节点任务影响多种设备  
                affected_equipment["node"] = task["node_id"]  
                affected_equipment["lines"] = task["equipment"]["lines"]  
                affected_equipment["switches"] = task["equipment"]["switches"]  
                affected_equipment["trafos"] = task["equipment"]["trafos"]  
            else:  
                # 单设备任务  
                affected_equipment[task["element_type"]] = [task["element_id"]]  
            
            affected_equipment["task_id"] = task["task_id"]  
            affected_equipment["needs_transfer"] = task["needs_transfer"]  
            affected_equipment["affected_loads"] = task["affected_loads"]  
            
            maintenance_schedule[i].append(affected_equipment)  

# 创建负荷和发电采样  
print("准备负荷和发电量数据...")  

# 创建负荷掩码以区分不同类型的负荷  
try:  
    load_r_mask = net.load['name'].str.contains('R').astype(float)  
    load_ind_mask = net.load['name'].str.contains('CI').astype(float)  
    load_lv_mask = net.load['name'].str.contains('LV').astype(float)  
    load_mv_mask = net.load['name'].str.contains('MV').astype(float)  
except:  
    # 如果没有对应的名称模式，使用默认分类  
    print("无法根据名称区分负荷类型，使用默认分类...")  
    load_r_mask = pd.Series(1.0, index=net.load.index)  # 默认都是居民负荷  
    load_ind_mask = pd.Series(0.0, index=net.load.index)  
    load_lv_mask = pd.Series(0.0, index=net.load.index)  
    load_mv_mask = pd.Series(0.0, index=net.load.index)  

# 创建发电掩码以区分不同类型的发电  
try:  
    sgen_pv_mask = net.sgen['name'].str.contains('PV').astype(float)  
    sgen_wind_mask = net.sgen['name'].str.contains('WKA').astype(float)  
    sgen_static_mask = net.sgen['name'].str.contains('Static').astype(float)  
except:  
    # 如果没有对应的名称模式，使用默认分类  
    print("无法根据名称区分发电类型，使用默认分类...")  
    sgen_pv_mask = pd.Series(0.5, index=net.sgen.index)  # 默认50%是光伏  
    sgen_wind_mask = pd.Series(0.3, index=net.sgen.index)  # 30%是风电  
    sgen_static_mask = pd.Series(0.2, index=net.sgen.index)  # 20%是常规发电  

# 初始化数据存储  
load_p = pd.DataFrame(index=net.load.index, columns=range(TIME_STEPS))  
load_sgen = pd.DataFrame(index=net.sgen.index, columns=range(TIME_STEPS))  

# 生成每个时间步的负荷和发电量  
for t in range(TIME_STEPS):  
    hour_of_day = date_range[t].hour  
    day_type_factor = weekend_factor[t]  
    seasonal_factor = season_factor[t]  
    
    # 计算负荷  
    load_p[t] = (load_r_mask + load_lv_mask) * net.load['p_mw'].mul(household_load_profile[hour_of_day]) * day_type_factor * seasonal_factor + \
               (load_ind_mask + load_mv_mask) * net.load['p_mw'].mul(industry_load_profile[hour_of_day]) * day_type_factor * seasonal_factor  
    
    # 计算发电量  
    if not net.sgen.empty:  
        load_sgen[t] = (sgen_pv_mask + sgen_static_mask) * net.sgen['p_mw'].mul(profile_day_sun[hour_of_day]) * seasonal_factor + \
                      sgen_wind_mask * net.sgen['p_mw'].mul(profile_day_wind[hour_of_day])  

# 添加随机性  
if LOAD_DIST == "normal":  
    # 添加负荷波动  
    load_noise = np.random.normal(1.0, 0.05, load_p.shape)  # 5%的随机波动  
    load_p = load_p * load_noise  
    
    # 添加发电波动  
    if not net.sgen.empty:  
        sgen_noise = np.random.normal(1.0, 0.08, load_sgen.shape)  # 8%的随机波动  
        load_sgen = load_sgen * sgen_noise  

# 初始化功率流结果存储  
pf_vm = pd.DataFrame(columns=range(TIME_STEPS), index=net.bus.index)  
pf_va = pd.DataFrame(columns=range(TIME_STEPS), index=net.bus.index)  
pf_p = pd.DataFrame(columns=range(TIME_STEPS), index=net.bus.index)  
pf_q = pd.DataFrame(columns=range(TIME_STEPS), index=net.bus.index)  

pf_pl = pd.DataFrame(columns=range(TIME_STEPS), index=net.line.index)  
pf_ql = pd.DataFrame(columns=range(TIME_STEPS), index=net.line.index)  
pf_loading = pd.DataFrame(columns=range(TIME_STEPS), index=net.line.index)  

# 存储检修前网络状态备份  
net_backup = net.deepcopy()  

print(f"开始执行{TIME_STEPS}个时间步的潮流计算...")  

# 主循环：对每个时间步进行潮流计算  
for t in tqdm(range(TIME_STEPS)):  
    # 恢复网络原始状态  
    net = copy.deepcopy(net_backup) 
    
    # 应用检修计划 - 改变设备状态  
    if t in maintenance_schedule:  
        for task in maintenance_schedule[t]:  
            # 处理不同类型的设备检修  
            if "node" in task:  
                # 节点检修 - 断开相关设备  
                node_id = task["node"]  
                
                # 断开连接到该节点的线路  
                for line_id in task.get("lines", []):  
                    if line_id in net.line.index:  
                        # 标记线路为停运  
                        net.line.loc[line_id, "in_service"] = False  
                
                # 断开与节点相关的开关  
                for switch_id in task.get("switches", []):  
                    if switch_id in net.switch.index:  
                        net.switch.loc[switch_id, "closed"] = False  
                
                # 断开变压器  
                for trafo_id in task.get("trafos", []):  
                    if trafo_id in net.trafo.index:  
                        net.trafo.loc[trafo_id, "in_service"] = False  
            
            # 单设备检修  
            elif "line" in task:  
                for line_id in task.get("line", []):  
                    if line_id in net.line.index:  
                        net.line.loc[line_id, "in_service"] = False  
            
            elif "switch" in task:  
                for switch_id in task.get("switch", []):  
                    if switch_id in net.switch.index:  
                        net.switch.loc[switch_id, "closed"] = False  
            
            elif "trafo" in task:  
                for trafo_id in task.get("trafo", []):  
                    if trafo_id in net.trafo.index:  
                        net.trafo.loc[trafo_id, "in_service"] = False  
            
            # 如果需要转供，模拟负荷转移  
            if task.get("needs_transfer", False):  
                for load_id in task.get("affected_loads", []):  
                    if load_id in net.load.index:  
                        # 模拟通过转供实现的负荷降低（转供后负荷可能会部分恢复）  
                        original_p = load_p.loc[load_id, t]  
                        # 假设转供可以恢复70%的负荷  
                        adjusted_p = original_p * 0.7  
                        load_p.loc[load_id, t] = adjusted_p  
    
    # 更新本时间步的负荷和发电值  
    net.load['p_mw'] = load_p[t]  
    net.load['q_mvar'] = load_p[t] * POWER_COEF  # 功率因数  
    if not net.sgen.empty:  
        net.sgen['p_mw'] = load_sgen[t]  
    
    # 执行潮流计算  
    try:  
        pp.runpp(net, calculate_voltage_angles=True, numba=False)  
        
        # 存储结果  
        mask = net.bus['vn_kv'] < 110  
        net.res_bus.loc[mask, "va_degree"] += net.trafo["shift_degree"].values[0] 
        net.res_bus["va_rad"] = net.res_bus["va_degree"] * np.pi / 180  
        pf_va[t] = net.res_bus["va_rad"]  
        pf_p[t] = net.res_bus["p_mw"]  
        pf_q[t] = net.res_bus["q_mvar"]  
        
        pf_pl[t] = net.res_line["p_from_mw"]  
        pf_ql[t] = net.res_line["q_from_mvar"]  
        pf_loading[t] = net.res_line["loading_percent"]  
        
    except pp.powerflow.LoadflowNotConverged:  
        print(f"\n时间步 {t} ({date_str_range[t]}) 潮流不收敛，可能是由于检修造成的网络拓扑变化")  
        # 使用上一个时间步的结果填充  
        if t > 0:  
            pf_vm[t] = pf_vm[t-1]  
            pf_va[t] = pf_va[t-1]  
            pf_p[t] = pf_p[t-1]  
            pf_q[t] = pf_q[t-1]  
            pf_pl[t] = pf_pl[t-1]  
            pf_ql[t] = pf_ql[t-1]  
            pf_loading[t] = pf_loading[t-1]  
        else:  
            # 第一个时间步就不收敛，使用初始值  
            pf_vm[t] = 1.0  
            pf_va[t] = 0.0  
            pf_p[t] = 0.0  
            pf_q[t] = 0.0  
            pf_pl[t] = 0.0  
            pf_ql[t] = 0.0  
            pf_loading[t] = 0.0  

# 保存结果  
print("保存仿真结果...")  

# 保存潮流结果  
with open(f'data/{grid_name}/pf_vm.pkl', 'wb') as f:  
    pickle.dump(pf_vm, f)  
with open(f'data/{grid_name}/pf_va.pkl', 'wb') as f:  
    pickle.dump(pf_va, f)  
with open(f'data/{grid_name}/pf_p.pkl', 'wb') as f:  
    pickle.dump(pf_p, f)  
with open(f'data/{grid_name}/pf_q.pkl', 'wb') as f:  
    pickle.dump(pf_q, f)  
with open(f'data/{grid_name}/pf_pl.pkl', 'wb') as f:  
    pickle.dump(pf_pl, f)  
with open(f'data/{grid_name}/pf_ql.pkl', 'wb') as f:  
    pickle.dump(pf_ql, f)  
with open(f'data/{grid_name}/pf_loading.pkl', 'wb') as f:  
    pickle.dump(pf_loading, f)  

# 保存负荷和发电数据  
with open(f'data/{grid_name}/load_p.pkl', 'wb') as f:  
    pickle.dump(load_p, f)  
if not net.sgen.empty:  
    with open(f'data/{grid_name}/load_sgen.pkl', 'wb') as f:  
        pickle.dump(load_sgen, f)  

# 保存检修计划  
with open(f'data/{grid_name}/maintenance_tasks.pkl', 'wb') as f:  
    pickle.dump(maintenance_tasks, f)  
with open(f'data/{grid_name}/maintenance_schedule.pkl', 'wb') as f:  
    pickle.dump(maintenance_schedule, f)  

# 保存时间索引信息  
with open(f'data/{grid_name}/time_index.pkl', 'wb') as f:  
    pickle.dump(date_str_range, f)  

# 生成检修计划Excel报表  
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
                "任务描述": t["desc"]  
            }  
        else:  # 单设备任务  
            row = {  
                "任务ID": t["task_id"],  
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
                "任务描述": t["desc"]  
            }  
        data.append(row)  
    
    # 创建DataFrame并保存为Excel  
    df = pd.DataFrame(data)  
    df.to_excel(filename, index=False)  
    return filename  

# 生成Excel报表  
excel_file = create_excel_report(maintenance_tasks, f'data/{grid_name}/maintenance_plan.xlsx')  
print(f"检修计划Excel报表已生成: {excel_file}")  

# 生成检修任务统计信息  
task_stats = {  
    "总任务数": len(maintenance_tasks),  
    "节点级任务": sum(1 for t in maintenance_tasks if t["task_type"] == "node"),  
    "单设备任务": sum(1 for t in maintenance_tasks if t["task_type"] == "single"),  
    "需要转供任务": sum(1 for t in maintenance_tasks if t.get("needs_transfer", False)),  
    "高优先级任务": sum(1 for t in maintenance_tasks if t["priority"] == 1),  
    "中优先级任务": sum(1 for t in maintenance_tasks if t["priority"] == 2),  
    "低优先级任务": sum(1 for t in maintenance_tasks if t["priority"] == 3)  
}  

# 生成基本的检修任务统计报告  
with open(f'data/{grid_name}/task_statistics.txt', 'w') as f:  
    f.write("检修任务统计信息\n")  
    f.write("=" * 30 + "\n")  
    for key, value in task_stats.items():  
        f.write(f"{key}: {value}\n")  
    f.write("\n检修任务时间分布\n")  
    f.write("=" * 30 + "\n")  
    
    # 按日期分组统计任务数量  
    task_dates = {}  
    for task in maintenance_tasks:  
        date = task["start_time"].split(" ")[0]  
        if date not in task_dates:  
            task_dates[date] = 0  
        task_dates[date] += 1  
    
    for date, count in sorted(task_dates.items()):  
        f.write(f"{date}: {count}个任务\n")  

print("仿真完成! 所有结果已保存到 data/" + grid_name + " 目录")  