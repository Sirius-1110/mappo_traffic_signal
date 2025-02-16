import unittest
import torch
import torch.nn as nn
from easydict import EasyDict
from typing import Union, Sequence, Dict, Any, Optional
from itertools import product
from ding.model.template.mavac_sota import MAVACSota, FCEncoder, ConvEncoder, RegressionHead, DiscreteHead, ReparameterizationHead, AgentEncoder, ClsEncoder, SelfAttention, CrosAttention

# Helper function to simulate squeeze
def squeeze(shape):
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        return shape[0]
    return shape

class TestMAVACSota(unittest.TestCase):

    def setUp(self):
        # 创建一个 MAVACSota 实例，用于测试
        cfg = EasyDict({
            'agent_obs_shape': 64,
            'global_obs_shape': 128,
            'action_shape': 10,
            'agent_num': 5,
            'actor_head_hidden_size': 256,
            'actor_head_layer_num': 2,
            'critic_head_hidden_size': 512,
            'critic_head_layer_num': 1,
            'action_space': 'discrete',
            'activation': nn.ReLU(),
            'norm_type': None,
            'sigma_type': 'independent',
            'bound_type': None,
        })
        self.model = MAVACSota(**cfg)

    def test_sequential_actor_head(self):
        # 测试 actor_head 是否正确
        x = torch.randn(4, 64)  # 4 个样本，每个样本的观测维度为 64
        output = self.model.actor_head(x)
        self.assertEqual(output['logit'].shape, torch.Size([4, 10]))  # 10 个动作

    def test_sequential_critic_head(self):
        # 测试 critic_head 是否正确
        x = torch.randn(4, 132)  # 4 个样本，每个样本的观测维度为 132 (128 + 4)
        output = self.model.critic_head(x)
        self.assertEqual(output['pred'].shape, torch.Size([4, 1]))  # 1 个 Q 值

    def test_compute_actor_discrete(self):
        # 测试 compute_actor 在离散动作空间中的行为
        inputs = {
            'agent_state': torch.randn(4, 64),
            'action_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
        }
        output = self.model.compute_actor(inputs)
        self.assertEqual(output['logit'].shape, torch.Size([4, 10]))  # 10 个动作
        self.assertTrue(torch.isinf(output['logit'][1, 9]))  # 动作 9 被掩码
        self.assertTrue(torch.isinf(output['logit'][2, 8]))  # 动作 8 和 9 被掩码
        self.assertTrue(torch.isinf(output['logit'][2, 9]))
        self.assertTrue(torch.isinf(output['logit'][3, 7]))  # 动作 7, 8, 9 被掩码
        self.assertTrue(torch.isinf(output['logit'][3, 8]))
        self.assertTrue(torch.isinf(output['logit'][3, 9]))

    def test_compute_actor_continuous(self):
        # 测试 compute_actor 在连续动作空间中的行为
        self.model.action_space = 'continuous'
        inputs = {
            'agent_state': torch.randn(4, 64)
        }
        output = self.model.compute_actor(inputs)
        self.assertEqual(output['logit'].shape, torch.Size([4, 10]))  # 10 个动作

    def test_compute_critic(self):
        # 测试 compute_critic 的行为
        inputs = {
            'agent_state': torch.randn(4, 64, 5),  # 4 个样本，每个样本有 5 个智能体，每个智能体的观测维度为 64
            'cls_state': torch.randn(4, 2, 64, 10),  # 4 个样本，每个样本有 2 个聚类，每个聚类有 64 个智能体，每个智能体的观测维度为 10
            'global_state': torch.randn(4, 128)  # 4 个样本，每个样本的全局观测维度为 128
        }
        output = self.model.compute_critic(inputs)
        self.assertEqual(output['value'].shape, torch.Size([4, 1]))  # 1 个 Q 值

    def test_compute_actor_critic(self):
        # 测试 compute_actor_critic 的行为
        inputs = {
            'agent_state': torch.randn(4, 64, 5),  # 4 个样本，每个样本有 5 个智能体，每个智能体的观测维度为 64
            'cls_state': torch.randn(4, 2, 64, 10),  # 4 个样本，每个样本有 2 个聚类，每个聚类有 64 个智能体，每个智能体的观测维度为 10
            'global_state': torch.randn(4, 128)  # 4 个样本，每个样本的全局观测维度为 128
        }
        output = self.model.compute_actor_critic(inputs)
        self.assertEqual(output['logit'].shape, torch.Size([4, 10]))  # 10 个动作
        self.assertEqual(output['value'].shape, torch.Size([4, 1]))  # 1 个 Q 值

if __name__ == '__main__':
    unittest.main()