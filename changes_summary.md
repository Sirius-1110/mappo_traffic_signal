# MAPPO-TSC 代码变更汇总

> 生成日期：2026-02-09
> 项目路径：`marl_sigctrl/`

本文档记录了针对 MAPPO 多交叉口信号控制项目的全部代码修改，涵盖配置修复、模型实现修正、环境接口修复和训练入口适配四个方面，共涉及 **9 个文件**。

---

## 一、评估频率配置修复（4 个文件）

### 问题描述

所有实验配置中 `eval_freq` 被设置为 `1e100`（即 $10^{100}$），导致训练循环中 `evaluator.should_eval(train_iter)` 几乎永远不会触发。训练期间仅在 `train_iter=0` 时执行一次评估，后续迭代不再产生评估记录，使得 evaluator 日志和 TensorBoard 曲线无法反映训练趋势，容易被误判为"模型不收敛"。

### 修改内容

将 `eval_freq` 从 `1e100` 改为 `1000`，即每 1000 次训练迭代触发一次评估。

### 涉及文件

#### 1. `signal_control/entry/sumo_config/sumo_3roads_mappo_baseline.py`

```python
# 修改前
eval=dict(evaluator=dict(eval_freq=1e100, ))

# 修改后
eval=dict(evaluator=dict(eval_freq=1000, ))
```

#### 2. `signal_control/entry/sumo_config/sumo_3roads_mappo_sota.py`

```python
# 修改前
eval=dict(evaluator=dict(eval_freq=1e100, ))

# 修改后
eval=dict(evaluator=dict(eval_freq=1000, ))
```

#### 3. `signal_control/entry/sumo_config/sumo_7roads_mappo_baseline.py`

```python
# 修改前
eval=dict(evaluator=dict(eval_freq=1e100, ))

# 修改后
eval=dict(evaluator=dict(eval_freq=1000, ))
```

#### 4. `signal_control/entry/sumo_config/sumo_7roads_mappo_sota.py`

```python
# 修改前
eval=dict(evaluator=dict(eval_freq=1e100, ))

# 修改后
eval=dict(evaluator=dict(eval_freq=1000, ))
```

---

## 二、Cross-Attention 实现修正（2 个文件）

### 问题描述

SOTA 模型（`MAVACSota` 和 `MAVAC`）中的 `CrosAttention` 模块存在语义错误：原实现将 cluster embedding 作为 query、agent embedding 作为 key/value，这与"让每个 agent 关注全局聚类信息"的设计意图相反。正确的做法是 agent embedding 作为 query（提问者），cluster embedding 作为 key/value（被查询的知识库），使得每个 agent 能够从聚类特征中提取与自身相关的全局信息。

此外，`mavac.py` 的 sota 分支还存在两个额外 bug：一个冗余的变量计算 `clsnt_emb`，以及 critic_head 的输入错误地传入了原始 obs dict 而非注意力融合后的张量。

### 涉及文件

#### 5. `framework/ding/model/template/mavac_sota.py`

**修改 A — CrosAttention.forward（第 170–182 行）：**

```python
# 修改前
def forward(self, agent_state, integrate_cluster_emb):
    T, B, A, N = agent_state.shape
    query = self.query(integrate_cluster_emb)   # cluster 作 query（错误）
    key = self.key(agent_state)                  # agent 作 key（错误）
    value = self.value(agent_state)              # agent 作 value（错误）
    attention_map = torch.matmul(query, key.transpose(-1, -2))  # T, B, K, A
    ...
    return cross_attention_output  # T, B, K, hidden_dim（错误形状）

# 修改后
def forward(self, agent_state, integrate_cluster_emb):
    T, B, A, N = agent_state.shape
    query = self.query(agent_state)              # agent 作 query（正确）
    key = self.key(integrate_cluster_emb)        # cluster 作 key（正确）
    value = self.value(integrate_cluster_emb)    # cluster 作 value（正确）
    attention_map = torch.matmul(query, key.transpose(-1, -2))  # T, B, A, K
    ...
    return cross_attention_output  # T, B, A, hidden_dim（正确形状）
```

**修改 B — compute_critic 调用顺序（第 461 行）：**

```python
# 修改前
integrate_cls_emb = self._cros_attn(cls_emb, agent_emb)

# 修改后
integrate_cls_emb = self._cros_attn(agent_emb, cls_emb)
```

**修改 C — 删除残留 print（原第 462 行）：**

```python
# 删除
print("cls_emb shape ", cls_emb.shape, "agent_emb shape ", agent_emb.shape)
```

#### 6. `framework/ding/model/template/mavac.py`

**修改 A — CrosAttention.forward（第 169–182 行）：**

与 `mavac_sota.py` 相同的 query/key/value 方向修正。

**修改 B — compute_critic sota 分支（第 437–443 行）：**

```python
# 修改前
agent_emb = self._agent_encoder(agent_state)
clsnt_emb = self._agent_encoder(agent_state)   # 冗余变量，从未使用
cls_emb = self._cls_encoder(cls_state, edges)
integrate_obs_emb = self._self_attn(agent_emb)
integrate_cls_emb = self._cros_attn(cls_emb, agent_emb)  # 参数顺序错误
integrate_emb = torch.add(integrate_obs_emb, integrate_cls_emb)
x = self.critic_head(x)  # 传入原始 dict，会报错

# 修改后
agent_emb = self._agent_encoder(agent_state)
cls_emb = self._cls_encoder(cls_state, edges)
integrate_obs_emb = self._self_attn(agent_emb)
integrate_cls_emb = self._cros_attn(agent_emb, cls_emb)  # 参数顺序正确
integrate_emb = torch.add(integrate_obs_emb, integrate_cls_emb)
x = self.critic_encoder(torch.concat([integrate_emb, global_state], dim=-1))
x = self.critic_head(x)  # 传入融合后的张量
```

---

## 三、环境 Seed 修复（1 个文件）

### 问题描述

`SumoEnv.seed()` 方法中，传入的 `seed` 和 `dynamic_seed` 参数被硬编码覆盖为 0，导致命令行传入的 `--seed` 参数不起作用，多环境并行采样时随机过程被锁死，实验可复现性和数据多样性均受影响。

#### 7. `signal_control/smartcross/envs/sumo_env.py`

```python
# 修改前
def seed(self, seed: int, dynamic_seed: bool = True) -> None:
    seed = 0           # 强制覆盖（错误）
    dynamic_seed = 0   # 强制覆盖（错误）
    self._seed = seed
    self._dynamic_seed = dynamic_seed
    np.random.seed(self._seed)

# 修改后
def seed(self, seed: int, dynamic_seed: bool = True) -> None:
    self._seed = seed
    self._dynamic_seed = dynamic_seed
    np.random.seed(self._seed)
    random.seed(self._seed)
```

---

## 四、PPO 策略残留调试语句清理（1 个文件）

#### 8. `framework/ding/policy/ppo.py`

```python
# 删除原第 144 行
print("check", self._cfg.learn.grad_clip_value, self._cfg.learn.grad_clip_type)
```

---

## 五、训练入口路径适配（1 个文件）

### 问题描述

`sumo_train` 脚本中硬编码了 Linux 绝对路径 `/home/cidi/mappo_traffic_signal/`，在不同部署环境下无法运行。

#### 9. `signal_control/entry/sumo_train`

```python
# 修改前
sys.path.append(r"/home/cidi/mappo_traffic_signal/")

# 修改后
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
for _sub in [_PROJECT_ROOT,
             os.path.join(_PROJECT_ROOT, 'framework'),
             os.path.join(_PROJECT_ROOT, 'signal_control')]:
    if _sub not in sys.path:
        sys.path.insert(0, _sub)
```

---

## 六、变更汇总表

| 序号 | 文件 | 变更类型 | 简述 |
| ---- | ---- | -------- | ---- |
| 1 | `sumo_3roads_mappo_baseline.py` | 配置 | `eval_freq`: 1e100 → 1000 |
| 2 | `sumo_3roads_mappo_sota.py` | 配置 | `eval_freq`: 1e100 → 1000 |
| 3 | `sumo_7roads_mappo_baseline.py` | 配置 | `eval_freq`: 1e100 → 1000 |
| 4 | `sumo_7roads_mappo_sota.py` | 配置 | `eval_freq`: 1e100 → 1000 |
| 5 | `mavac_sota.py` | 模型 | cross-attention 方向修正 + 删 print |
| 6 | `mavac.py` | 模型 | cross-attention 修正 + 删冗余变量 + 修 critic 输入 |
| 7 | `sumo_env.py` | 环境 | seed 不再硬编码为 0 |
| 8 | `ppo.py` | 策略 | 删残留 print |
| 9 | `sumo_train` | 入口 | 路径改为动态计算 |
