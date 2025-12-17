data_gen_utils.py 用于生成“可信”的负荷与光伏时序（带空间相关性）
grid_utils.py 用于拓扑体检 + 自动补一个联络开关（Tie Switch）

env/：环境类（GridMaintenanceEnv）

tasks/：任务生成/读取/标准化

topology/：联络开关 map、校验、动作方案生成

scripts/：每阶段的验证脚本

logs/：验证脚本输出