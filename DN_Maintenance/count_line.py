import pandapower.networks as pn  
import pandapower.plotting as plot  

net = pn.mv_oberrhein(include_substations=True)  

# 去掉 show_plot 参数  
plot.simple_plotly(net)  