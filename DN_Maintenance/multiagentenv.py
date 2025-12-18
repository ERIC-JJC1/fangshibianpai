class MultiAgentEnv:
    def __init__(self, **kwargs):
        pass

    def step(self, actions):
        """
        执行动作并返回结果
        :param actions: List of actions for each agent
        :return: obs, rewards, dones, infos
        """
        raise NotImplementedError

    def get_obs(self):
        """
        返回每个智能体的观测
        :return: List of observations for each agent
        """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """
        返回特定智能体的观测
        :param agent_id: Agent index
        :return: Observation for the agent
        """
        raise NotImplementedError

    def get_obs_size(self):
        """
        返回观测空间大小
        :return: Observation size
        """
        raise NotImplementedError

    def get_state(self):
        """
        返回全局状态
        :return: Global state
        """
        raise NotImplementedError

    def get_state_size(self):
        """
        返回全局状态大小
        :return: State size
        """
        raise NotImplementedError

    def get_avail_actions(self):
        """
        返回每个智能体可用的动作
        :return: Available actions for each agent
        """
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """
        返回特定智能体可用的动作
        :param agent_id: Agent index
        :return: Available actions for the agent
        """
        raise NotImplementedError

    def get_total_actions(self):
        """
        返回动作空间大小
        :return: Action space size
        """
        raise NotImplementedError

    def reset(self):
        """
        重置环境
        :return: Initial observations and states
        """
        raise NotImplementedError

    def render(self):
        """
        渲染环境（可选）
        """
        pass

    def close(self):
        """
        关闭环境（可选）
        """
        pass

    def seed(self, seed):
        """
        设置随机种子
        :param seed: Random seed
        """
        pass

    def save_replay(self):
        """
        保存回放（可选）
        """
        pass

    def get_env_info(self):
        """
        返回环境信息
        :return: Environment information dictionary
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info