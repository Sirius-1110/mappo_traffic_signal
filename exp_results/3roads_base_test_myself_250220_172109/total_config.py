exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 2,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'renew',
            'retry_waiting_time': 0.1,
            'shared_memory': False,
            'copy_on_get': True,
            'context': 'spawn',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 0,
        'n_evaluator_episode': 1,
        'import_names': ['smartcross.envs.sumo_env'],
        'type': 'sumo_env',
        'sumocfg_path': 'sumo_3roads/rl_wj.sumocfg',
        'gui': False,
        'inference': False,
        'max_episode_steps': 1500,
        'green_duration': 10,
        'yellow_duration': 3,
        'tls': ['ftddj_wjj', 'ftddj_frj', 'htddj_gsndj'],
        'obs': {
            'obs_type': ['phase', 'lane_pos_vec'],
            'lane_grid_num': 10,
            'traffic_volumn_ratio': 7.5,
            'use_centralized_obs': False,
            'padding': True
        },
        'action': {
            'action_type': 'change',
            'use_multi_discrete': True
        },
        'reward': {
            'use_centralized_reward': True,
            'reward_type': {
                'queue_len': 1.0
            }
        },
        'collector_env_num': 1,
        'evaluator_env_num': 1,
        'agent_num': 3,
        'obs_dim': 174,
        'cls_num': 1
    },
    'policy': {
        'model': {
            'agent_obs_shape': 174,
            'global_obs_shape': 442,
            'action_shape': 4,
            'agent_num': 3,
            'activation': Tanh()
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 1000,
                    'save_ckpt_after_iter': 1000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'epoch_per_collect': 10,
            'batch_size': 256,
            'learning_rate': 0.0001,
            'value_weight': 0.5,
            'entropy_weight': 0.01,
            'clip_ratio': 0.2,
            'adv_norm': True,
            'value_norm': True,
            'ppo_param_init': True,
            'grad_clip_type': 'clip_norm',
            'grad_clip_value': 0.5,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': True,
                'collect_print_freq': 10,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'discount_factor': 0.6,
            'gae_lambda': 0.95,
            'n_sample': 600
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1e+100,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 0,
                'n_episode': 1
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 4096,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': True,
        'cuda': True,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'type': 'ppo',
        'priority': False,
        'priority_IS_weight': False,
        'recompute_adv': True,
        'action_space': 'discrete',
        'nstep_return': False,
        'multi_agent': True,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOPolicyDict',
        'import_names': ['ding.policy.ppo'],
        'sota': False
    },
    'exp_name': 'mappo_3roads_base_test_myself_250220_172109',
    'seed': None
}
