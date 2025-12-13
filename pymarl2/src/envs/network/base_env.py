"""
独立的多智能体环境基类，避免PyMARL2导入冲突
"""

class MultiAgentEnv:
    """多智能体环境基类"""
    
    def __init__(self):
        pass
    
    def step(self, actions):
        """执行动作"""
        raise NotImplementedError
    
    def reset(self):
        """重置环境"""
        raise NotImplementedError
    
    def get_obs(self):
        """获取观测"""
        raise NotImplementedError
    
    def get_state(self):
        """获取全局状态"""
        raise NotImplementedError
    
    def get_avail_actions(self):
        """获取可用动作"""
        return None
    
    def get_env_info(self):
        """获取环境信息"""
        raise NotImplementedError
    
    def render(self):
        """渲染环境"""
        pass
    
    def close(self):
        """关闭环境"""
        pass