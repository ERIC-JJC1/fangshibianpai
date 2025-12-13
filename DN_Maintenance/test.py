import pandapower.networks as pn

# 假设你已有 read_tasks_from_excel、GridMaintenanceEnv 的定义在 env.py
from task_env import read_tasks_from_excel, GridMaintenanceEnv

net = pn.mv_oberrhein()
tasks = read_tasks_from_excel('maintenance_plan_20250521_151713.xlsx')
forecast_loads = None  # 传None或空dict，如果你暂时不用预测负荷

env = GridMaintenanceEnv(net, tasks, forecast_loads)
obs, state = env.reset()
print('[check] obs shape:', [o.shape for o in obs])
print('[check] state shape:', state.shape)

# ------- 随机动作模拟一步 -------
actions = []
for aid in range(env.agent_num):
    region_tasks = [t for t in env.tasks if t['region_id'] == aid and t['status'] == 'unassigned']
    agent_actions = []
    for t in region_tasks:
        # -- 选第一个合法时间 --
        start = 0  # 你也可以用random.randint(...)取第一个可用
        duration = t['duration']
        transfer = 0  # 可以改为随机可选的转供编号，如 random.choice(可选idx)
        agent_actions.append([start, duration, transfer])
    actions.append(agent_actions)

obs, reward, done, info = env.step(actions)
print('[check] reward:', reward)
print('[check] info:', info)
print('[check] obs:', obs)
print('[check] done:', done)