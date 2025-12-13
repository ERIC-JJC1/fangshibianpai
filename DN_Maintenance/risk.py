import numpy as np
import matplotlib.pyplot as plt

def simulate_reward_curve(num_episodes=400):
    rewards = []
    initial_penalty = -1000  # 初始惩罚值
    current_penalty = initial_penalty

    for episode in range(num_episodes):
        if episode < 100:
            # 在前 100 回合，随机波动
            current_penalty += np.random.uniform(-20, 10)  # 在这个阶段大幅度震荡
        elif 100 <= episode < 200:
            # 在 100 到 200 回合之间减少负值，保持波动
            current_penalty += np.random.uniform(-10, 20)  # 随机幅度减小，仍有波动
        elif 200 <= episode < 300:
            # 在 200 和 300 回合之间保持较小波动，进一步收敛
            current_penalty += np.random.uniform(-5, 15)  # 继续减小负值并增加小幅波动
        else:
            # 300 次后，震荡目标值附近
            current_penalty += np.random.uniform(-3, 3)  # 在接近 -20 的范围内波动

        # 限制奖励不低于 -20
        current_penalty = min(current_penalty, -20)
        # 将奖励加入列表
        rewards.append(current_penalty)

    return rewards

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Simulated Reward per Episode', color='blue')
    plt.title('Simulated Reward Curve with Random Oscillation')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.axhline(0, color='red', linestyle='--', label='Zero Reward')  # y=0的参考线
    plt.axhline(-20, color='green', linestyle='--', label='Convergence Level (-20)')  # 收敛水平
    plt.ylim(-2000, 0)  # 设置y轴范围，便于观察
    plt.legend()
    plt.grid()
    plt.savefig("simulated_rl_random_oscillation_curve.png")  # 保存图像文件
    plt.show()

# 运行模拟
rewards = simulate_reward_curve()
plot_rewards(rewards)