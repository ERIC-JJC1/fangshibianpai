import pandapower.networks as pn  
import pandapower as pp  

# 加载Oberrhein网络  
net = pn.mv_oberrhein()  

# 打印所有部件的DataFrame名字（即net的主要表结构）  
print("所有部件表：", net.keys())  

# 查看所有负荷的母线编号和备注  
print(net.load[['bus', 'name']])  

# 或者查bus表  
print(net.bus[['name']])
# 逐个显示元件的条数与主要字段  
for table in ['bus', 'line', 'trafo', 'switch', 'load', 'sgen', 'gen', 'shunt', 'impedance']:  
    df = net[table]  
    print(f"\n【{table}】数量: {len(df)}")  
    print(df.head())  


# 统计不同电压等级bus的数量  
print(net.bus['vn_kv'].value_counts())  

# 网络的所有分区、变电站、馈线编号（如有region/group等字段）  
if 'zone' in net.bus.columns:  
    print(net.bus['zone'].value_counts())  

# 每个bus上的负荷数量  
print(net.load.groupby("bus").size().sort_values(ascending=False).head())  

# 输出典型线路数据  
print(net.line[['from_bus','to_bus','length_km','type']].head())  