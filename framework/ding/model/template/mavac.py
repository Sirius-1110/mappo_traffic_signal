from typing import Union, Dict, Optional
import torch
import torch.nn as nn
from ding.torch_utils.network.nn_module import fc_block
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder
import math 
import itertools
from torch_geometric.nn import GCNConv
from typing import Union, List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, squeeze, MODEL_REGISTRY
from ding.torch_utils.network.nn_module import fc_block, MLP
from ding.torch_utils.network.transformer import ScaledDotProductAttention
from ding.torch_utils import to_tensor, tensor_to_list
from .q_learning import DRQN
import math 
import itertools
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, 
            input_dim, 
            hidden_dim, 
            output_dim
        ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x 
    
class GraphWeight(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim: int = 8, 
                 output_dim: int = 8
        ):
        super(GraphWeight, self).__init__()
        self._hidden_dim = hidden_dim
        self.q = GCN(input_dim, hidden_dim, output_dim)
        self.k = GCN(input_dim, hidden_dim, output_dim)
        return 
    def remove_diag(self, attention_map):
        T, B, A, _ = attention_map.shape
        mask = torch.ones(attention_map.shape, dtype=torch.bool, device=attention_map.device)
        for i in range(A):
            mask[:, :, i, i] = False
        attention_map = attention_map[mask]
        
        return attention_map.reshape(T, B, A, A-1)
    def forward(self, agent_state, edges):
        # >>> agent_state = (T, B, A, N)
        T, B, A, N = agent_state.shape
        query = self.q(agent_state, edges)
        key = self.k(agent_state, edges)
        key = key.reshape(T, B, self._hidden_dim, A)
        attention_map = torch.matmul(query, key)  # T, B, A, hidden_dim
        attention_map /= math.sqrt(1)
        attention_map = F.softmax(attention_map, dim=-1)
        return self.remove_diag(attention_map)    # T, B, A, A-1

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim = 8):
        super(Attention, self).__init__()
        self._hidden_dim = hidden_dim
        self.q = fc_block(input_dim, hidden_dim)
        self.k = fc_block(input_dim, hidden_dim)
    def remove_diag(self, attention_map):
        T, B, A, _ = attention_map.shape
        mask = torch.ones(attention_map.shape, dtype=torch.bool, device=attention_map.device)
        for i in range(A):
            mask[:, :, i, i] = False
        attention_map = attention_map[mask]
        return attention_map.reshape(T, B, A, A-1)
    def forward(self, agent_state):
        # >>> agent_state = (T, B, A, N)
        T, B, A, N = agent_state.shape
        query = self.q(agent_state)
        key = self.k(agent_state)
        key = key.reshape(T, B, self._hidden_dim, A)
        attention_map = torch.matmul(query, key)  # T, B, A, hidden_dim
        attention_map /= math.sqrt(1)
        attention_map = F.softmax(attention_map, dim=-1)
        return self.remove_diag(attention_map)    # T, B, A, A-1

class GraphEncoder(nn.Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()
        return 
    def forward(self, attention_map, agent_relation):
        # >>> attention_map = (T, B, A, A-1)
        # >>> agent_relation = (T, B, A, A-1 * 2)
        T, B, A, _ = agent_relation.shape
        agent_relation = agent_relation.reshape(T, B, A, (A-1), 2)
        attention_map = attention_map.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # (1, 1, A, A-1, 1)
        weight_agent_relation = attention_map * agent_relation
        
        return weight_agent_relation.reshape(T, B, A, -1)

class AgentEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim: int = 4
        ):
        super(AgentEncoder, self).__init__()
        self.encoder = fc_block(input_dim, hidden_dim)
        return 

    def forward(self, agent_state):
        return self.encoder(agent_state)

class ClsEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim: int = 4, 
                 output_dim: int = 4
        ):
        super(ClsEncoder, self).__init__()
        self._hidden_dim = hidden_dim
        self.encoder = GCN(input_dim, hidden_dim, output_dim)
        return 

    def forward(self, neigborhood_state, edges):
        neighborhood_emb = self.encoder(neigborhood_state, edges)
        neighborhood_emb = torch.mean(neighborhood_emb, dim=-2, keepdim=False)  
        
        return neighborhood_emb
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4):
        super(SelfAttention, self).__init__()
        self._hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, agent_state):
        T, B, A, N = agent_state.shape

        # Self-attention
        query = self.query(agent_state)
        key = self.key(agent_state)
        value = self.value(agent_state)
        
        attention_map = torch.matmul(query, key.transpose(-1, -2))  # T, B, A, A
        attention_map /= math.sqrt(self._hidden_dim)
        attention_map = F.softmax(attention_map, dim=-1)
        self_attention_output = torch.matmul(attention_map, value)  # T, B, A, hidden_dim
        
        return self_attention_output  # T, B, A, hidden_dim

class CrosAttention(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4):
        super(CrosAttention, self).__init__()
        self._hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, agent_state, integrate_cluster_emb):
        T, B, A, N = agent_state.shape

        # Cross-attention
        query = self.query(agent_state)
        key = self.key(integrate_cluster_emb)
        value = self.value(integrate_cluster_emb)
        
        attention_map = torch.matmul(query, key.transpose(-1, -2))  # T, B, A, K
        attention_map /= math.sqrt(self._hidden_dim)
        attention_map = F.softmax(attention_map, dim=-1)
        cross_attention_output = torch.matmul(attention_map, value)  # T, B, A, hidden_dim
        
        return cross_attention_output  # T, B, A, hidden_dim

@MODEL_REGISTRY.register('mavac')
class MAVAC(nn.Module):
    r"""
    Overview:
        The MAVAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
        self,
        agent_obs_shape: Union[int, SequenceType],
        global_obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType],
        agent_num: int,
        actor_head_hidden_size: int = 256,
        actor_head_layer_num: int = 2,
        critic_head_hidden_size: int = 512,
        critic_head_layer_num: int = 1,
        action_space: str = 'discrete',
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        sigma_type: Optional[str] = 'independent',
        bound_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the VAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - share_encoder (:obj:`bool`): Whether share encoder.
            - continuous (:obj:`bool`): Whether collect continuously.
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details`
        """
        super(MAVAC, self).__init__()
        self.sota = False
        agent_obs_shape: int = squeeze(agent_obs_shape)
        global_obs_shape: int = squeeze(global_obs_shape)
        action_shape: int = squeeze(action_shape)
        self.global_obs_shape, self.agent_obs_shape, self.action_shape = global_obs_shape, agent_obs_shape, action_shape
        self.action_space = action_space
        # Encoder Type
        if isinstance(agent_obs_shape, int) or len(agent_obs_shape) == 1:
            encoder_cls = FCEncoder
        elif len(agent_obs_shape) == 3:
            encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(agent_obs_shape)
            )
        if isinstance(global_obs_shape, int) or len(global_obs_shape) == 1:
            global_encoder_cls = FCEncoder
        elif len(global_obs_shape) == 3:
            global_encoder_cls = ConvEncoder
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".
                format(global_obs_shape)
            )

        # We directly connect the Head after a Liner layer instead of using the 3-layer FCEncoder.
        # In SMAC task it can obviously improve the performance.
        # Users can change the model according to their own needs.
        self.actor_encoder = nn.Identity()
        self.critic_encoder = nn.Identity()
        # Head Type
        if self.sota:
            global_obs_shape = global_obs_shape + 4
        self.critic_head = nn.Sequential(
            nn.Linear(global_obs_shape, critic_head_hidden_size), activation,
            RegressionHead(
                critic_head_hidden_size, 1, critic_head_layer_num, activation=activation, norm_type=norm_type
            )
        )
        assert self.action_space in ['discrete', 'continuous'], self.action_space
        if self.action_space == 'discrete':
            self.actor_head = nn.Sequential(
                nn.Linear(agent_obs_shape, actor_head_hidden_size), activation,
                DiscreteHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.action_space == 'continuous':
            self.actor_head = nn.Sequential(
                nn.Linear(agent_obs_shape, actor_head_hidden_size), activation,
                ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type=sigma_type,
                    activation=activation,
                    norm_type=norm_type,
                    bound_type=bound_type
                )
            )
        # must use list, not nn.ModuleList
        self.actor = [self.actor_encoder, self.actor_head]
        self.critic = [self.critic_encoder, self.critic_head]
        # for convenience of call some apis(such as: self.critic.parameters()), but may cause
        # misunderstanding when print(self)
        self.actor = nn.ModuleList(self.actor)
        self.critic = nn.ModuleList(self.critic)

        # Transformer && Graph neural netowrk)
        self._agent_encoder = AgentEncoder(agent_obs_shape)
        self._cls_encoder = ClsEncoder(agent_obs_shape)
        self._self_attn = SelfAttention()
        self._cros_attn = CrosAttention()

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Parameter updates with VAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'`` or ``'compute_critic'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Forward with ``'compute_actor'``, Necessary Keys:
                    - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N corresponding ``hidden_size``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Actor Examples:
            >>> model = VAC(64,128)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['logit'].shape == torch.Size([4, 128])

        Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)

        Actor-Critic Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, x: torch.Tensor) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
        """
        if self.action_space == 'discrete':
            action_mask = x['action_mask']
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x['logit']
            logit[action_mask == 0.0] = -99999999
        elif self.action_space == 'continuous':
            x = x['agent_state']
            x = self.actor_encoder(x)
            x = self.actor_head(x)
            logit = x
        return {'logit': logit}

    def compute_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`Dict`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = critic_head_hidden_size``
        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

                Necessary Keys:
                    - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> critic_outputs = model(inputs,'compute_critic')
            >>> critic_outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
        """
        if self.sota:
            agent_state = x['agent_state']
            cls_state = x['cls_state']
            global_state = x['global_state']
            
            single_step = len(agent_state.shape) == 3
            if single_step:
                agent_state = agent_state.unsqueeze(0)
                cls_state = cls_state.unsqueeze(0)
                global_state = global_state.unsqueeze(0)

            T, B, A = agent_state.shape[:3]
            edges = torch.tensor(list(itertools.combinations(range(A), 2))).t().contiguous().to(agent_state.device)

            agent_emb = self._agent_encoder(agent_state)
            cls_emb = self._cls_encoder(cls_state, edges)
            integrate_obs_emb = self._self_attn(agent_emb)
            integrate_cls_emb = self._cros_attn(agent_emb, cls_emb)
            integrate_emb = torch.add(integrate_obs_emb, integrate_cls_emb)
            x = self.critic_encoder(torch.concat([integrate_emb, global_state], dim=-1))
            x = self.critic_head(x)
            if single_step:
                x['pred'] = x['pred'].squeeze(0)
        else:
            x = self.critic_encoder(x['global_state'])
            x = self.critic_head(x)
        return {'value': x['pred']}

    def compute_actor_critic(self, x: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_actor_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The encoded embedding tensor.

        Returns:
            - outputs (:obj:`Dict`):
                Run with encoder and head.

        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Logit encoding tensor, with same size as input ``x``.
            - value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is ``action_shape``
            - value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> model = VAC(64,64)
            >>> inputs = torch.randn(4, 64)
            >>> outputs = model(inputs,'compute_actor_critic')
            >>> outputs['value']
            tensor([0.0252, 0.0235, 0.0201, 0.0072], grad_fn=<SqueezeBackward1>)
            >>> assert outputs['logit'].shape == torch.Size([4, 64])


        .. note::
            ``compute_actor_critic`` interface aims to save computation when shares encoder.
            Returning the combination dictionry.

        """
        logit = self.compute_actor(x)['logit']
        value = self.compute_critic(x)['value']
        return {'logit': logit, 'value': value}
