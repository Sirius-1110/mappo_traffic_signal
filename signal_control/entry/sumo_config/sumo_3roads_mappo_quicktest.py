from easydict import EasyDict
from torch import nn

sumo_mappo_default_config = dict(
    exp_name=f'quicktest_3roads_baseline',
    env=dict(
        manager=dict(
            shared_memory=False,
            context='spawn',
            retry_type='renew',
            max_retry=2,
        ),
        n_evaluator_episode=1,
        stop_value=0,
        collector_env_num=1,
        evaluator_env_num=1,
        agent_num=3,
        obs_dim=174,
        cls_num=1
    ),
    policy=dict(
        sota=False,
        cuda=False,
        priority=False,
        multi_agent=True,
        action_space='discrete',
        model=dict(
            agent_obs_shape=174,
            global_obs_shape=442,
            action_shape=4,
            agent_num=3,
            activation=nn.Tanh(),
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=1e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(
                train_iterations=10,
                hook=dict(
                    save_ckpt_after_iter=5,
                    log_show_after_iter=1,
                ),
            ),
            value_norm=True,
        ),
        collect=dict(
            unroll_len=1,
            discount_factor=0.6,
            gae_lambda=0.95,
            n_sample=100,
            collector=dict(
                transform_obs=True,
                collect_print_freq=1,
            )
        ),
        eval=dict(evaluator=dict(eval_freq=5, )),
        other=dict()
    ),
)

create_config = dict(
    env_manager=dict(
        type='subprocess',
    ),
    env=dict(
        import_names=['smartcross.envs.sumo_env'],
        type='sumo_env',
    ),
    policy=dict(
        import_names=['ding.policy.ppo'],
        type='ppo',
    ),
)

create_config = EasyDict(create_config)
sumo_mappo_default_config = EasyDict(sumo_mappo_default_config)
main_config = sumo_mappo_default_config
