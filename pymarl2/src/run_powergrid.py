#!/usr/bin/env python3

"""
配电网检修调度多智能体强化学习训练脚本
完整修复版本
"""

import datetime
import os
import sys
import threading
from types import SimpleNamespace as SN

import torch as th

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run(_run, _config, _log):
    """主运行函数"""
    
    # 将字典转换为SimpleNamespace对象，解决PyMARL兼容性问题
    if isinstance(_config, dict):
        config_obj = SN()
        for key, value in _config.items():
            setattr(config_obj, key, value)
    else:
        config_obj = _config
    
    # 检查CUDA可用性
    use_cuda = getattr(config_obj, 'use_cuda', True) and th.cuda.is_available()
    setattr(config_obj, 'use_cuda', use_cuda)
    device = "cuda" if use_cuda else "cpu"
    _log.info(f"使用设备: {device}")
        # ✅ 将device设置到config_obj中
    setattr(config_obj, 'device', device)
    # 设置随机种子
    import numpy as np
    import random
    seed = getattr(config_obj, 'seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    

    required_defaults = {
        # 核心组件
        'agent_output_type': 'q',
        'action_selector': 'epsilon_greedy',
        'agent': 'rnn',
        'mac': 'basic_mac',
        'learner': 'q_learner',
        'mixer': 'qmix',
        'runner': 'episode',
        
        # 网络参数
        'hidden_dim': 64,
        'rnn_hidden_dim': 64,
        'mixing_embed_dim': 32,
        'hypernet_layers': 2,
        'hypernet_embed': 64,
        
        # ✅ 优化器参数
        'optimizer': 'adam',  # 优化器类型
        'lr': 0.0005,
        'critic_lr': 0.0005,
        'agent_lr': 0.0005,
        'mixer_lr': 0.0005,
        'optim_alpha': 0.99,
        'optim_eps': 0.00001,
        'optim_weight_decay': 0,
        'grad_norm_clip': 10,
        
        # 强化学习参数
        'gamma': 0.99,
        'td_lambda': 0.8,
        'double_q': True,
        'target_update_interval': 200,
        
        # 探索参数
        'epsilon_start': 1.0,
        'epsilon_finish': 0.05,
        'epsilon_anneal_time': 50000,
        'epsilon_decay': 'linear',
        'test_epsilon': 0.0,
        'evaluation_epsilon': 0.0,
        
        # 训练参数
        'batch_size': 32,
        'batch_size_run': 1,
        'buffer_size': 32000,
        'buffer_cpu_only': True,
        'learning_starts': 1000,
        'train_freq': 4,
        'train_batch_size': 32,
        'use_replay_buffer': True,
        'replay_buffer_size': 32000,
        
        
        # 观测参数
        'obs_last_action': True,
        'obs_agent_id': True,
        'obs_individual_obs': False,
        'obs_instead_of_state': False,
        'state_last_action': True,
        
        # 设备和路径参数
        'device': device,
        'local_results_path': 'results',
        'unique_token': 'powergrid_experiment',
        'label': 'powergrid_qmix',
        'log_train_stats_t': -1,
        'log_eval_stats_t': -1,
        'last_log_T': 0,
        'model_save_time': 0,
        
        # ✅ 添加更多可能需要的参数
        'standardise_returns': False,  # 标准化回报
        'standardise_rewards': False,  # 标准化奖励
        'use_rnn': True,  # 使用RNN
        'obs_own_feats_only': False,  # 只观测自己的特征
        'obs_timestep_number': False,  # 观测时间步
        'mask_before_softmax': True,  # softmax前进行mask
        'hard_qs': False,  # 硬Q值
        'mixer_norm': 'abs',  # 混合网络归一化
        'num_atoms': 1,  # 原子数量（分布式RL）
        'v_min': 0.0,  # 最小值（分布式RL）
        'v_max': 20.0,  # 最大值（分布式RL）
        
        # 其他参数
        'num_workers': 1,
        'render': False,
        'record': False,
        'use_tensorboard': False,  # 是否使用tensorboard
        'save_replay': False,  # 是否保存回放
        'test_greedy': True,  # 测试时使用贪心策略
        'test_nepisode': 32,  # 测试episode数量
        'test_interval': 10000,  # 测试间隔
        'log_interval': 2000,  # 日志间隔
        'runner_log_interval': 2000,  # 运行器日志间隔
        'learner_log_interval': 2000,  # 学习器日志间隔
        't_max': 2000000,  # 最大训练步数
        'save_model': True,  # 是否保存模型
        'save_model_interval': 100000,  # 模型保存间隔
        'checkpoint_path': '',  # 检查点路径
        'load_step': 0,  # 加载步数
        'evaluate': False,  # 是否评估
        'seed': 12345,  # 随机种子
    }
    
    # 设置缺失的默认参数
    missing_params = []
    for param, default_value in required_defaults.items():
        if not hasattr(config_obj, param):
            setattr(config_obj, param, default_value)
            missing_params.append(param)
    
    if missing_params:
        _log.info(f"设置了 {len(missing_params)} 个默认参数: {missing_params[:10]}...")
            
    # 导入PyMARL组件
    from envs import REGISTRY as env_REGISTRY
    from learners import REGISTRY as le_REGISTRY
    from runners import REGISTRY as r_REGISTRY
    from controllers import REGISTRY as mac_REGISTRY
    from components.episode_buffer import ReplayBuffer
    from components.transforms import OneHot
    
    # 检查环境是否注册
    env_name = getattr(config_obj, 'env', 'powergrid')
    if env_name not in env_REGISTRY:
        available_envs = list(env_REGISTRY.keys())
        _log.error(f"环境 '{env_name}' 未注册！")
        _log.error(f"可用环境: {available_envs}")
        raise ValueError(f"环境 '{env_name}' 未注册")
    
    # 创建环境
    _log.info(f"创建环境: {env_name}")
    env_args = getattr(config_obj, 'env_args', {})
    _log.info(f"环境参数: {env_args}")
    
    try:
        env = env_REGISTRY[env_name](**env_args)
        _log.info("环境创建成功")
    except Exception as e:
        _log.error(f"环境创建失败: {e}")
        raise
    
    # 获取环境信息并更新配置
    env_info = env.get_env_info()
    
    # 将环境信息添加到config_obj
    for key, value in env_info.items():
        setattr(config_obj, key, value)
    
    _log.info(f"环境信息: {env_info}")
    
    # 设置数据格式
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    
    groups = {"agents": env_info["n_agents"]}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])}
    
    # 创建缓冲区
    buffer_size = getattr(config_obj, 'buffer_size', 32000)
    buffer_cpu_only = getattr(config_obj, 'buffer_cpu_only', True)
    
    buffer = ReplayBuffer(
        scheme, groups, buffer_size, 
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if buffer_cpu_only else device
    )
    
    _log.info(f"缓冲区创建成功: 大小={buffer_size}, CPU-only={buffer_cpu_only}")
    
    # 设置多智能体控制器
    mac_name = getattr(config_obj, 'mac', 'basic_mac')
    try:
        mac = mac_REGISTRY[mac_name](buffer.scheme,groups,config_obj)
        _log.info(f"多智能体控制器创建成功: {mac_name}")
    except Exception as e:
        _log.error(f"多智能体控制器创建失败: {e}")
        raise
    
    # 设置运行器
    runner_name = getattr(config_obj, 'runner', 'episode')
    try:
        runner = r_REGISTRY[runner_name](args=config_obj,logger=_log)
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        _log.info(f"运行器创建成功: {runner_name}")
    except Exception as e:
        _log.error(f"运行器创建失败: {e}")
        raise
    
    # 设置学习器
    learner_name = getattr(config_obj, 'learner', 'q_learner')
    try:
        learner = le_REGISTRY[learner_name](mac,buffer.scheme,_log,config_obj)
        _log.info(f"学习器创建成功: {learner_name}")
    except Exception as e:
        _log.error(f"学习器创建失败: {e}")
        raise
    
    if getattr(config_obj, 'use_cuda', False):
        learner.cuda()
        _log.info("学习器已移动到GPU")
    
    # 检查是否从检查点恢复
    checkpoint_path = getattr(config_obj, 'checkpoint_path', "")
    if checkpoint_path and checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0
        if not os.path.isdir(checkpoint_path):
            _log.warning(f"检查点路径不存在: {checkpoint_path}")
            setattr(config_obj, 'checkpoint_path', "")
        else:
            # 寻找最新检查点
            for name in os.listdir(checkpoint_path):
                full_name = os.path.join(checkpoint_path, name)
                if os.path.isdir(full_name):
                    try:
                        timestep = int(name)
                        timesteps.append(timestep)
                    except ValueError:
                        continue
            
            load_step = getattr(config_obj, 'load_step', 0)
            if load_step != 0:
                timestep_to_load = load_step
            elif timesteps:
                timestep_to_load = max(timesteps)
            
            if timestep_to_load > 0:
                model_path = os.path.join(checkpoint_path, str(timestep_to_load))
                _log.info(f"从检查点加载模型: {model_path}")
                try:
                    learner.load_models(model_path)
                    runner.t_env = timestep_to_load
                    _log.info(f"成功加载检查点: t_env={timestep_to_load}")
                except Exception as e:
                    _log.error(f"加载检查点失败: {e}")
    
    # 训练参数
    t_max = getattr(config_obj, 't_max', 2000000)
    test_interval = getattr(config_obj, 'test_interval', 10000)
    test_nepisode = getattr(config_obj, 'test_nepisode', 32)
    log_interval = getattr(config_obj, 'log_interval', 2000)
    save_model = getattr(config_obj, 'save_model', False)
    save_model_interval = getattr(config_obj, 'save_model_interval', 100000)
    batch_size = getattr(config_obj, 'batch_size', 32)
    
    # 开始训练
    episode = 0
    last_test_T = -test_interval - 1
    last_log_T = 0
    model_save_time = 0
    
    start_time = datetime.datetime.now()
    last_time = start_time
    
    _log.info("="*60)
    _log.info("开始训练...")
    _log.info(f"最大训练步数: {t_max}")
    _log.info(f"测试间隔: {test_interval}")
    _log.info(f"日志间隔: {log_interval}")
    _log.info(f"批次大小: {batch_size}")
    _log.info("="*60)
    
    try:
        while runner.t_env <= t_max:
            # 运行一个episode
            try:
                episode_batch = runner.run(test_mode=False)
                buffer.insert_episode_batch(episode_batch)
                
                # 训练
                if buffer.can_sample(batch_size):
                    episode_sample = buffer.sample(batch_size)
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    
                    if episode_sample.device != device:
                        episode_sample.to(device)
                    
                    learner.train(episode_sample, runner.t_env, episode)
                
            except Exception as e:
                _log.error(f"Episode {episode} 运行失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 执行测试
            if (runner.t_env - last_test_T) / test_interval >= 1.0:
                _log.info(f"进行测试 - t_env: {runner.t_env} / {t_max}")
                
                # 计算剩余时间
                current_time = datetime.datetime.now()
                elapsed = (current_time - start_time).total_seconds()
                if runner.t_env > 0:
                    estimated_total = elapsed * t_max / runner.t_env
                    remaining = estimated_total - elapsed
                    remaining_str = str(datetime.timedelta(seconds=int(remaining)))
                    _log.info(f"预计剩余时间: {remaining_str}")
                
                last_test_T = runner.t_env
                
                # 运行测试episodes
                try:
                    for test_ep in range(test_nepisode):
                        runner.run(test_mode=True)
                        if test_ep % 8 == 0:  # 每8个测试episode报告一次进度
                            _log.info(f"测试进度: {test_ep+1}/{test_nepisode}")
                except Exception as e:
                    _log.error(f"测试过程出错: {e}")
            
            # 保存模型
            if save_model and (runner.t_env - model_save_time >= save_model_interval):
                model_save_time = runner.t_env
                
                results_path = getattr(config_obj, 'local_results_path', 'results')
                unique_token = getattr(config_obj, 'unique_token', 'default')
                save_path = os.path.join(results_path, "models", unique_token, str(runner.t_env))
                
                try:
                    os.makedirs(save_path, exist_ok=True)
                    _log.info(f"保存模型到: {save_path}")
                    learner.save_models(save_path)
                    _log.info("模型保存成功")
                except Exception as e:
                    _log.error(f"模型保存失败: {e}")
            
            episode += 1
            
            # 定期日志
            if (runner.t_env - last_log_T) >= log_interval:
                progress = runner.t_env / t_max * 100
                _log.info(f"训练进度: Episode {episode}, t_env {runner.t_env}/{t_max} ({progress:.1f}%)")
                last_log_T = runner.t_env
                
    except KeyboardInterrupt:
        _log.info("训练被用户中断")
    except Exception as e:
        _log.error(f"训练过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            runner.close_env()
            _log.info("环境已关闭")
        except:
            pass
        
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        _log.info(f"训练总时间: {total_time}")
        _log.info("训练完成！")

def main():
    """主函数"""
    
    # 加载配置文件
    config_path = "src/config/default_powergrid.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("请确保在项目根目录运行此脚本")
        print("当前工作目录:", os.getcwd())
        sys.exit(1)
    
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)
    
    # 验证关键配置
    required_fields = ['env']
    for field in required_fields:
        if field not in config:
            print(f"错误：配置文件中缺少必需字段 '{field}'")
            sys.exit(1)
    
    # 设置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(asctime)s] %(name)s %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger("root")
    
    # 创建结果目录
    results_path = "results/powergrid_experiment"
    os.makedirs(results_path, exist_ok=True)
    config["local_results_path"] = results_path
    
    # 检查CUDA可用性
    cuda_available = th.cuda.is_available()
    use_cuda = config.get('use_cuda', False) and cuda_available
    
    print("=" * 60)
    print("     配电网检修调度多智能体强化学习训练")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"环境: {config.get('env', 'Unknown')}")
    print(f"结果路径: {results_path}")
    print(f"CUDA可用: {cuda_available}")
    print(f"使用CUDA: {use_cuda}")
    print(f"PyTorch版本: {th.__version__}")
    if cuda_available:
        print(f"CUDA设备数量: {th.cuda.device_count()}")
        print(f"当前CUDA设备: {th.cuda.current_device()}")
    print("=" * 60)
    
    # 检查环境参数文件是否存在
    env_args = config.get('env_args', {})
    for file_key in ['task_file', 'load_curve_file']:
        if file_key in env_args:
            file_path = env_args[file_key]
            if not os.path.exists(file_path):
                print(f"警告: {file_key} 文件不存在: {file_path}")
            else:
                print(f"✓ {file_key}: {file_path}")
    
    print("=" * 60)
    
    # 开始训练
    try:
        run(None, config, logger)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()