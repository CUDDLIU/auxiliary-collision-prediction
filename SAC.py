import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"
min_Val = torch.tensor(1e-7, dtype=torch.float32, device=device)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if x.dim() == 3:
            return x + self.pe[:, : x.size(1), :]
        return x + self.pe[: x.size(0), :]


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=src_mask)
        x = self.ln1(x + attn_output)
        ff_output = self.ff(x)
        x = self.ln2(x + ff_output)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_blocks, n_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(embed_dim, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.enc_dec_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = self.ln1(x + self_attn_output)
        enc_dec_attn_output, _ = self.enc_dec_attention(x, enc_output, enc_output, attn_mask=memory_mask)
        x = self.ln2(x + enc_dec_attn_output)
        ff_output = self.ff(x)
        x = self.ln3(x + ff_output)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_blocks, n_heads, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, input_len, embed_dim, n_blocks, n_heads, dropout):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = Encoder(embed_dim, n_blocks, n_heads, dropout)
        self.decoder = Decoder(embed_dim, n_blocks, n_heads, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        return dec_output


class LidarStateAdapter(nn.Module):
    """LiRA-SAC 的模型侧观测适配器。

    环境仍然输出原始的 72 维 LiDAR + 机器人状态向量。LiRA-SAC 只在网络
    内部改变表示方式，因此训练和评估接口仍能兼容 prior Transformer-SAC/CTSAC。
    """

    def __init__(self, state_dim, lidar_repr="raw", lidar_dim=72, lidar_max_range=10.0, collision_dist=0.35):
        super(LidarStateAdapter, self).__init__()
        self.state_dim = state_dim
        self.lidar_repr = lidar_repr
        self.lidar_dim = lidar_dim
        self.robot_dim = state_dim - lidar_dim
        self.lidar_max_range = float(lidar_max_range)
        self.collision_dist = float(collision_dist)
        if lidar_repr == "raw":
            self.output_dim = state_dim
        elif lidar_repr == "multi_v1":
            self.output_dim = self.lidar_dim * 3 + self.robot_dim
        else:
            raise ValueError("Unsupported lidar_repr: {}".format(lidar_repr))

    def forward(self, state):
        scan = state[..., : self.lidar_dim]
        robot = state[..., self.lidar_dim :]
        if self.lidar_repr == "raw":
            return torch.cat((scan, robot), dim=-1)

        # LiRA-SAC 创新点 1：确定性的风险感知 LiDAR 重参数化。
        # raw_norm 保留几何距离信息，inv_norm 放大近障碍风险，abs_grad 突出
        # 障碍边界/通道开口等距离突变。这里不引入额外可学习感知模块，
        # 但比单纯 raw range 更直接地把安全相关结构提供给 actor/critic。
        raw_norm = torch.clamp(scan, 0.0, self.lidar_max_range) / self.lidar_max_range
        denom = torch.clamp(scan, min=self.collision_dist, max=self.lidar_max_range)
        inv_norm = torch.clamp(self.collision_dist / denom, 0.0, 1.0)
        grad_prev = torch.roll(raw_norm, shifts=1, dims=-1)
        grad_next = torch.roll(raw_norm, shifts=-1, dims=-1)
        abs_grad = 0.5 * torch.abs(grad_next - grad_prev)
        lidar_features = torch.cat((raw_norm, inv_norm, abs_grad), dim=-1)
        return torch.cat((lidar_features, robot), dim=-1)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, temporal_backbone="transformer", embed_dim=1024, n_blocks=2, n_heads=8, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        self.temporal_backbone = temporal_backbone
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        if temporal_backbone == "transformer":
            self.temporal = Transformer(input_len=embed_dim, embed_dim=embed_dim, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout)
        elif temporal_backbone == "gru":
            self.temporal = nn.GRU(embed_dim, embed_dim, batch_first=True)
        else:
            raise ValueError("Unsupported temporal_backbone: {}".format(temporal_backbone))

    def _project(self, x):
        return F.relu(self.fc1(x))

    def forward(self, x, eval=False, src_mask=None, tgt_mask=None, memory_mask=None):
        x = self._project(x)
        if self.temporal_backbone == "transformer":
            # 时序骨干沿用 prior Transformer-SAC 思路：把当前投影 token 与
            # 最近状态序列中的时序上下文结合。LiRA-SAC 不把新 Transformer
            # 结构作为主要创新，重点放在 LiDAR 风险表示和碰撞辅助监督。
            x_ctx = self.temporal(x, x, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if eval:
                return F.relu(self.fc2(torch.cat((x, x_ctx), dim=2)))
            return F.relu(self.fc2(torch.cat((x[:, -1, :], x_ctx[:, -1, :]), dim=1)))

        x_ctx, _ = self.temporal(x)
        if eval:
            return F.relu(self.fc2(torch.cat((x[:, -1:, :], x_ctx[:, -1:, :]), dim=2)))
        return F.relu(self.fc2(torch.cat((x[:, -1, :], x_ctx[:, -1, :]), dim=1)))


class LiRASACActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        lidar_repr="raw",
        lidar_max_range=10.0,
        temporal_backbone="transformer",
        min_log_std=-10,
        max_log_std=2,
        n_blocks=2,
        n_heads=8,
        embed_dim=1024,
        dropout=0.1,
    ):
        super(LiRASACActor, self).__init__()
        self.adapter = LidarStateAdapter(state_dim, lidar_repr=lidar_repr, lidar_max_range=lidar_max_range)
        self.temporal_encoder = TemporalEncoder(
            self.adapter.output_dim,
            temporal_backbone=temporal_backbone,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.mu_head = nn.Linear(embed_dim, action_dim)
        self.log_std_head = nn.Linear(embed_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, obs_state, eval=False, src_mask=None, tgt_mask=None, memory_mask=None):
        adapted_state = self.adapter(obs_state)
        features = self.temporal_encoder(
            adapted_state,
            eval=eval,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        mu = self.mu_head(features)
        log_std_head = self.log_std_head(features)
        return mu, torch.clamp(log_std_head, self.min_log_std, self.max_log_std)


class LiRASACCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        lidar_repr="raw",
        lidar_max_range=10.0,
        temporal_backbone="transformer",
        n_blocks=2,
        n_heads=8,
        embed_dim=1024,
        dropout=0.1,
    ):
        super(LiRASACCritic, self).__init__()
        self.adapter = LidarStateAdapter(state_dim, lidar_repr=lidar_repr, lidar_max_range=lidar_max_range)
        self.temporal_encoder = TemporalEncoder(
            self.adapter.output_dim,
            temporal_backbone=temporal_backbone,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, 1)

    def forward(self, obs_state, src_mask=None, tgt_mask=None, memory_mask=None):
        adapted_state = self.adapter(obs_state)
        features = self.temporal_encoder(
            adapted_state,
            eval=False,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        hidden = F.relu(self.fc3(features))
        return self.fc4(hidden)


class MonotonicMultiHorizonRiskHead(nn.Module):
    """Predict cumulative collision risk with p(H1) <= p(H2) <= ... ."""

    def __init__(self, embed_dim, horizon_count):
        super(MonotonicMultiHorizonRiskHead, self).__init__()
        if horizon_count < 1:
            raise ValueError("horizon_count must be positive")
        self.projection = nn.Linear(embed_dim, horizon_count)

    def forward(self, hidden):
        raw = self.projection(hidden)
        first = raw[..., :1]
        if raw.size(-1) == 1:
            logits = first
        else:
            logits = torch.cat((first, first + torch.cumsum(F.softplus(raw[..., 1:]), dim=-1)), dim=-1)
        return logits, torch.sigmoid(logits)


class IndependentMultiHorizonRiskHead(nn.Module):
    """Predict each cumulative collision horizon without ordering constraints."""

    def __init__(self, embed_dim, horizon_count):
        super(IndependentMultiHorizonRiskHead, self).__init__()
        if horizon_count < 1:
            raise ValueError("horizon_count must be positive")
        self.projection = nn.Linear(embed_dim, horizon_count)

    def forward(self, hidden):
        logits = self.projection(hidden)
        return logits, torch.sigmoid(logits)


def balanced_binary_cross_entropy_with_logits(logits, targets, masks=None):
    """Average positive and negative BCE equally within each horizon."""

    if logits.shape != targets.shape:
        raise ValueError("risk logits and targets must have the same shape")
    if masks is None:
        masks = torch.ones_like(targets)
    if masks.shape != targets.shape:
        raise ValueError("risk masks and targets must have the same shape")
    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    horizon_losses = []
    for horizon_index in range(logits.size(-1)):
        horizon_targets = targets[..., horizon_index]
        horizon_element_loss = element_loss[..., horizon_index]
        valid = masks[..., horizon_index] >= 0.5
        positive = (horizon_targets >= 0.5) & valid
        negative = (horizon_targets < 0.5) & valid
        class_losses = []
        if positive.any():
            class_losses.append(horizon_element_loss[positive].mean())
        if negative.any():
            class_losses.append(horizon_element_loss[negative].mean())
        if class_losses:
            horizon_losses.append(torch.stack(class_losses).mean())
    if not horizon_losses:
        return logits.sum() * 0.0
    return torch.stack(horizon_losses).mean()


class LiRASACQ(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        lidar_repr="raw",
        lidar_max_range=10.0,
        temporal_backbone="transformer",
        use_collision_aux=False,
        use_multi_horizon_risk=False,
        risk_horizon_count=3,
        risk_head_mode="monotonic",
        n_blocks=2,
        n_heads=8,
        embed_dim=1024,
        dropout=0.1,
    ):
        super(LiRASACQ, self).__init__()
        self.use_collision_aux = use_collision_aux
        self.use_multi_horizon_risk = use_multi_horizon_risk
        if self.use_collision_aux and self.use_multi_horizon_risk:
            raise ValueError("one-step collision aux and multi-horizon risk cannot be enabled together")
        self.adapter = LidarStateAdapter(state_dim, lidar_repr=lidar_repr, lidar_max_range=lidar_max_range)
        self.temporal_encoder = TemporalEncoder(
            self.adapter.output_dim + action_dim,
            temporal_backbone=temporal_backbone,
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, 1)
        # LiRA-SAC 创新点 2：只在 Q1 上挂接碰撞风险预测头。
        # 监督目标是采样 state-action 的下一步 collision label，使 critic 学到
        # 更安全感知的隐藏特征，同时保留标准 twin-Q SAC 的价值估计路径。
        self.collision_head = nn.Linear(embed_dim, 1) if use_collision_aux else None
        if use_multi_horizon_risk:
            risk_head_class = {
                "monotonic": MonotonicMultiHorizonRiskHead,
                "independent": IndependentMultiHorizonRiskHead,
            }.get(risk_head_mode)
            if risk_head_class is None:
                raise ValueError("unknown risk_head_mode: {}".format(risk_head_mode))
            self.risk_head = risk_head_class(embed_dim, risk_horizon_count)
        else:
            self.risk_head = None

    def forward(
        self,
        obs_state,
        a,
        return_aux=False,
        return_risk=False,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
    ):
        if a.dim() == 2:
            a = a.unsqueeze(1)
        if a.size(1) == 1 and obs_state.size(1) > 1:
            a = a.expand(-1, obs_state.size(1), -1)
        elif a.size(1) != obs_state.size(1):
            raise ValueError(
                "LiRASACQ.forward expects action sequence length to match state sequence length, got {} vs {}".format(
                    a.size(1), obs_state.size(1)
                )
            )

        adapted_state = self.adapter(obs_state)
        state_action = torch.cat((adapted_state, a), dim=2)
        features = self.temporal_encoder(
            state_action,
            eval=False,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        hidden = F.relu(self.fc3(features))
        q_value = self.fc4(hidden)
        if return_aux and self.collision_head is not None:
            return q_value, self.collision_head(hidden)
        if return_risk:
            if self.risk_head is None:
                raise RuntimeError("return_risk=True requires a multi-horizon risk head")
            risk_logits, risk_probabilities = self.risk_head(hidden)
            return q_value, risk_logits, risk_probabilities
        return q_value


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, args, paths, tb_dir):
        super(SAC, self).__init__()

        self.args = args
        self.paths = paths
        self.tb_dir = tb_dir
        if not hasattr(self.args, "key_num"):
            self.args.key_num = self.args.keys_num
        self.sequence_length = self.args.keys_num
        self.lidar_repr = self.args.lidar_repr
        self.temporal_backbone = self.args.temporal_backbone
        self.use_collision_aux = self.args.use_collision_aux
        self.collision_aux_critics = getattr(self.args, "collision_aux_critics", "q1")
        self.use_multi_horizon_risk = bool(getattr(self.args, "use_multi_horizon_risk", False))
        self.risk_head_mode = getattr(self.args, "risk_head_mode", "monotonic")
        self.risk_critic_mode = getattr(self.args, "risk_critic_mode", "twin")
        self.use_actor_risk = bool(getattr(self.args, "use_actor_risk", False))
        self.risk_horizons = tuple(getattr(self.args, "risk_horizons", [5, 10, 20]))
        if len(self.risk_horizons) != 3:
            raise ValueError("risk_horizons must contain exactly three horizons")
        self.risk_weights = 1.0 / torch.tensor(self.risk_horizons, dtype=torch.float32, device=device)
        self.risk_weights = self.risk_weights / self.risk_weights.sum()
        self.q_scale_ema = 1.0

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        common_kwargs = dict(
            lidar_repr=self.lidar_repr,
            lidar_max_range=self.args.lidar_max_range,
            temporal_backbone=self.temporal_backbone,
        )
        print("[SAC] init start device={} repr={} backbone={}".format(device, self.lidar_repr, self.temporal_backbone), flush=True)
        print("[SAC] build policy_net", flush=True)
        self.policy_net = LiRASACActor(self.state_dim, self.action_dim, **common_kwargs).to(device)
        print("[SAC] build value_net", flush=True)
        self.value_net = LiRASACCritic(self.state_dim, **common_kwargs).to(device)
        print("[SAC] build target_value_net", flush=True)
        self.Target_value_net = LiRASACCritic(self.state_dim, **common_kwargs).to(device)
        print("[SAC] build Q_net1", flush=True)
        self.Q_net1 = LiRASACQ(
            self.state_dim,
            self.action_dim,
            use_collision_aux=self.use_collision_aux,
            use_multi_horizon_risk=self.use_multi_horizon_risk,
            risk_horizon_count=len(self.risk_horizons),
            risk_head_mode=self.risk_head_mode,
            **common_kwargs,
        ).to(device)
        print("[SAC] build Q_net2", flush=True)
        self.Q_net2 = LiRASACQ(
            self.state_dim,
            self.action_dim,
            use_collision_aux=self.use_collision_aux and self.collision_aux_critics == "twin",
            use_multi_horizon_risk=self.use_multi_horizon_risk and self.risk_critic_mode == "twin",
            risk_horizon_count=len(self.risk_horizons),
            risk_head_mode=self.risk_head_mode,
            **common_kwargs,
        ).to(device)

        print("[SAC] build optimizers", flush=True)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=self.args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=self.args.learning_rate)

        self.num_training = 1
        self.initial_tem = self.args.initial_tem
        self.tem_decay_rate = self.args.tem_decay_rate
        self.last_map_level = 0
        self.num_training_map_level = 0
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        print("[SAC] summary writer ready", flush=True)

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()
        if self.use_collision_aux:
            pos_weight = torch.tensor([self.args.collision_aux_pos_weight], dtype=torch.float32, device=device)
            self.collision_aux_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

    def _prepare_state(self, state):
        if isinstance(state, torch.Tensor):
            obs_state = state.to(device)
        elif isinstance(state, list):
            seq = [np.array(s, dtype=np.float32) for s in state[-self.sequence_length :]]
            obs_state = torch.as_tensor(np.stack(seq, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            obs_state = torch.as_tensor(state, dtype=torch.float32, device=device)

        if obs_state.dim() == 1:
            obs_state = obs_state.unsqueeze(0).unsqueeze(0)
        elif obs_state.dim() == 2:
            obs_state = obs_state.unsqueeze(0)
        return obs_state

    def select_action(self, state, deterministic=False):
        obs_state = self._prepare_state(state)
        mu, log_sigma = self.policy_net(obs_state)
        if deterministic:
            action = torch.tanh(mu).detach().cpu().numpy()
            return action[0]
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action[0]

    def predict_risk(self, state, action):
        if not self.use_multi_horizon_risk:
            return None
        obs_state = self._prepare_state(state)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        q1_was_training = self.Q_net1.training
        q2_was_training = self.Q_net2.training
        self.Q_net1.eval()
        self.Q_net2.eval()
        try:
            with torch.no_grad():
                _, _, risk_q1 = self.Q_net1(obs_state, action_tensor, return_risk=True)
                aggregate_q1 = (risk_q1 * self.risk_weights).sum(dim=-1)
                if self.risk_critic_mode == "twin":
                    _, _, risk_q2 = self.Q_net2(obs_state, action_tensor, return_risk=True)
                    conservative_per_horizon = torch.max(risk_q1, risk_q2)
                    aggregate_q2 = (risk_q2 * self.risk_weights).sum(dim=-1)
                    aggregate = torch.max(aggregate_q1, aggregate_q2)
                else:
                    risk_q2 = None
                    conservative_per_horizon = risk_q1
                    aggregate = aggregate_q1
        finally:
            self.Q_net1.train(q1_was_training)
            self.Q_net2.train(q2_was_training)
        return {
            "q1": risk_q1[0].detach().cpu().numpy(),
            "q2": risk_q2[0].detach().cpu().numpy() if risk_q2 is not None else None,
            "conservative": conservative_per_horizon[0].detach().cpu().numpy(),
            "aggregate": float(aggregate[0].detach().cpu()),
        }

    def evaluate(self, obs_state):
        obs_state = self._prepare_state(obs_state)
        batch_mu, batch_log_sigma = self.policy_net(obs_state, eval=True)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)
        z = noise.sample(batch_mu.shape).to(device)
        action = torch.tanh(batch_mu + batch_sigma * z)
        log_prob = dist.log_prob(batch_mu + batch_sigma * z) - torch.log(1 - action.pow(2) + min_Val)
        log_prob = log_prob.sum(dim=2, keepdim=True)
        return action, log_prob, z, batch_mu, batch_log_sigma

    def _current_risk_eta(self):
        if not self.use_actor_risk or self.num_training <= self.args.risk_warmup_steps:
            return 0.0
        if self.args.risk_ramp_steps <= 0:
            return float(self.args.risk_actor_eta)
        progress = (self.num_training - self.args.risk_warmup_steps) / float(self.args.risk_ramp_steps)
        return float(self.args.risk_actor_eta) * min(1.0, max(0.0, progress))

    def _set_q_requires_grad(self, enabled):
        for q_network in (self.Q_net1, self.Q_net2):
            for parameter in q_network.parameters():
                parameter.requires_grad_(enabled)

    def update(self, replay_buffer, batch_size, current_map_level):
        for _ in range(self.args.gradient_steps):
            self.tem = self.initial_tem / (1.0 + self.tem_decay_rate * self.num_training_map_level)

            sampled = replay_buffer.sample_batch(
                batch_size,
                self.sequence_length,
                include_risk_targets=self.use_multi_horizon_risk,
            )
            if self.use_multi_horizon_risk:
                s_state, a, r, s_, d, _end, collision, risk_targets, risk_masks = sampled
            else:
                s_state, a, r, s_, d, _end, collision = sampled
                risk_targets = None
                risk_masks = None
            bn_s_state = self._prepare_state(s_state)
            bn_a = a.to(device)
            bn_r = r.to(device)
            bn_s_ = self._prepare_state(s_)
            bn_d = d.to(device)
            bn_collision = collision.to(device)
            bn_risk_targets = risk_targets.to(device) if risk_targets is not None else None
            bn_risk_masks = risk_masks.to(device) if risk_masks is not None else None

            bn_r_last = bn_r[:, -1, :]
            bn_d_last = bn_d[:, -1, :]
            collision_last = bn_collision[:, -1, :]
            risk_targets_last = bn_risk_targets[:, -1, :] if bn_risk_targets is not None else None
            risk_masks_last = bn_risk_masks[:, -1, :] if bn_risk_masks is not None else None

            target_value = self.Target_value_net(bn_s_)
            next_q_value = bn_r_last + (1 - bn_d_last) * self.args.gamma * target_value

            expected_value = self.value_net(bn_s_state)
            q1_output = self.Q_net1(
                bn_s_state,
                bn_a,
                return_aux=self.use_collision_aux,
                return_risk=self.use_multi_horizon_risk,
            )
            if self.use_multi_horizon_risk:
                expected_Q1, risk_logits_q1, risk_probabilities_q1 = q1_output
                if self.risk_critic_mode == "twin":
                    expected_Q2, risk_logits_q2, risk_probabilities_q2 = self.Q_net2(
                        bn_s_state,
                        bn_a,
                        return_risk=True,
                    )
                else:
                    expected_Q2 = self.Q_net2(bn_s_state, bn_a)
                    risk_logits_q2 = None
                    risk_probabilities_q2 = None
                collision_logits = None
            elif self.use_collision_aux:
                expected_Q1, collision_logits = q1_output
                if self.collision_aux_critics == "twin":
                    expected_Q2, collision_logits_q2 = self.Q_net2(bn_s_state, bn_a, return_aux=True)
                else:
                    expected_Q2 = self.Q_net2(bn_s_state, bn_a)
                    collision_logits_q2 = None
            else:
                expected_Q1 = q1_output
                collision_logits = None
                expected_Q2 = self.Q_net2(bn_s_state, bn_a)

            sample_action, log_prob, *_ = self.evaluate(bn_s_state)
            log_prob_last = log_prob[:, -1, :]
            expected_new_Q = torch.min(self.Q_net1(bn_s_state, sample_action), self.Q_net2(bn_s_state, sample_action))
            next_value = expected_new_Q - self.tem * log_prob_last

            V_loss = self.value_criterion(expected_value, next_value.detach()).mean()
            Q1_loss = self.Q1_criterion(expected_Q1, next_q_value.detach()).mean()
            Q2_loss = self.Q2_criterion(expected_Q2, next_q_value.detach()).mean()

            q_scale_batch = torch.min(expected_Q1.detach(), expected_Q2.detach()).abs().mean().item()
            self.q_scale_ema = 0.99 * self.q_scale_ema + 0.01 * q_scale_batch

            if self.use_multi_horizon_risk:
                risk_aux_loss_q1 = balanced_binary_cross_entropy_with_logits(
                    risk_logits_q1, risk_targets_last, risk_masks_last
                )
                Q1_total_loss = Q1_loss + self.args.risk_aux_weight * risk_aux_loss_q1
                if self.risk_critic_mode == "twin":
                    risk_aux_loss_q2 = balanced_binary_cross_entropy_with_logits(
                        risk_logits_q2, risk_targets_last, risk_masks_last
                    )
                    Q2_total_loss = Q2_loss + self.args.risk_aux_weight * risk_aux_loss_q2
                else:
                    risk_aux_loss_q2 = None
                    Q2_total_loss = Q2_loss
                self.writer.add_scalar("Risk/aux_loss_q1", risk_aux_loss_q1, global_step=self.num_training)
                if risk_aux_loss_q2 is not None:
                    self.writer.add_scalar("Risk/aux_loss_q2", risk_aux_loss_q2, global_step=self.num_training)
                for index, horizon in enumerate(self.risk_horizons):
                    self.writer.add_scalar(
                        "Risk/h{}_prob_q1".format(horizon),
                        risk_probabilities_q1[:, index].mean(),
                        global_step=self.num_training,
                    )
                    if risk_probabilities_q2 is not None:
                        self.writer.add_scalar(
                            "Risk/h{}_prob_q2".format(horizon),
                            risk_probabilities_q2[:, index].mean(),
                            global_step=self.num_training,
                        )
                    self.writer.add_scalar(
                        "Risk/h{}_label".format(horizon),
                        (
                            risk_targets_last[:, index] * risk_masks_last[:, index]
                        ).sum() / risk_masks_last[:, index].sum().clamp_min(1.0),
                        global_step=self.num_training,
                    )
                    self.writer.add_scalar(
                        "Risk/h{}_valid_fraction".format(horizon),
                        risk_masks_last[:, index].mean(),
                        global_step=self.num_training,
                    )
            elif self.use_collision_aux:
                # LiRA-SAC 创新点 3：碰撞辅助监督与 Q1 loss 联合优化。
                # 它不是执行阶段的硬安全过滤器，而是在训练阶段塑造 critic 特征，
                # 让“回报相近但即时碰撞风险更高”的动作能被更清楚地区分。
                collision_aux_loss = self.collision_aux_criterion(collision_logits, collision_last)
                Q1_total_loss = Q1_loss + self.args.collision_aux_weight * collision_aux_loss
                if collision_logits_q2 is not None:
                    collision_aux_loss_q2 = self.collision_aux_criterion(collision_logits_q2, collision_last)
                    Q2_total_loss = Q2_loss + self.args.collision_aux_weight * collision_aux_loss_q2
                    self.writer.add_scalar(
                        "Aux/collision_loss_q2", collision_aux_loss_q2, global_step=self.num_training
                    )
                else:
                    collision_aux_loss_q2 = None
                    Q2_total_loss = Q2_loss
                collision_prob = torch.sigmoid(collision_logits.detach())
                self.writer.add_scalar("Aux/collision_loss", collision_aux_loss, global_step=self.num_training)
                self.writer.add_scalar("Aux/collision_prob_mean", collision_prob.mean(), global_step=self.num_training)
                self.writer.add_scalar("Aux/collision_label_mean", collision_last.mean(), global_step=self.num_training)
            else:
                collision_aux_loss = None
                Q1_total_loss = Q1_loss
                Q2_total_loss = Q2_loss

            self.writer.add_scalar("Loss/V_loss", V_loss, global_step=self.num_training)
            self.writer.add_scalar("Loss/Q1_loss", Q1_loss, global_step=self.num_training)
            self.writer.add_scalar("Loss/Q2_loss", Q2_loss, global_step=self.num_training)

            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_total_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_total_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            if current_map_level != self.last_map_level:
                self.num_training_map_level = 0

            if self.num_training % self.args.policy_update_interval == 0:
                self.policy_optimizer.zero_grad()
                eval_state = bn_s_state.detach()
                self._set_q_requires_grad(False)
                try:
                    sample_action, log_prob, *_ = self.evaluate(eval_state)
                    log_prob_last = log_prob[:, -1, :]
                    if self.use_multi_horizon_risk:
                        q1_policy, _risk_logits_q1, risk_probabilities_q1 = self.Q_net1(
                            eval_state,
                            sample_action,
                            return_risk=True,
                        )
                        if self.risk_critic_mode == "twin":
                            q2_policy, _risk_logits_q2, risk_probabilities_q2 = self.Q_net2(
                                eval_state,
                                sample_action,
                                return_risk=True,
                            )
                        else:
                            q2_policy = self.Q_net2(eval_state, sample_action)
                            risk_probabilities_q2 = None
                        expected_new_Q = torch.min(q1_policy, q2_policy)
                        aggregate_risk_q1 = (risk_probabilities_q1 * self.risk_weights).sum(dim=-1, keepdim=True)
                        if risk_probabilities_q2 is not None:
                            aggregate_risk_q2 = (risk_probabilities_q2 * self.risk_weights).sum(dim=-1, keepdim=True)
                            conservative_risk = torch.max(aggregate_risk_q1, aggregate_risk_q2)
                        else:
                            conservative_risk = aggregate_risk_q1
                        risk_excess = F.relu(conservative_risk - self.args.risk_budget)
                        risk_eta = self._current_risk_eta()
                        actor_risk_penalty = risk_eta * self.q_scale_ema * risk_excess
                    else:
                        expected_new_Q = torch.min(
                            self.Q_net1(eval_state, sample_action),
                            self.Q_net2(eval_state, sample_action),
                        )
                        conservative_risk = None
                        risk_excess = None
                        risk_eta = 0.0
                        actor_risk_penalty = 0.0
                    pi_loss = (self.tem * log_prob_last - expected_new_Q + actor_risk_penalty).mean()
                    pi_loss.backward()
                finally:
                    self._set_q_requires_grad(True)
                self.writer.add_scalar("Loss/policy_loss", pi_loss, global_step=self.num_training)
                if conservative_risk is not None:
                    self.writer.add_scalar("Risk/actor_risk", conservative_risk.mean(), global_step=self.num_training)
                    self.writer.add_scalar("Risk/actor_excess", risk_excess.mean(), global_step=self.num_training)
                    self.writer.add_scalar("Risk/actor_eta", risk_eta, global_step=self.num_training)
                    self.writer.add_scalar("Risk/q_scale_ema", self.q_scale_ema, global_step=self.num_training)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

            if self.num_training % self.args.target_update_interval == 0:
                for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(target_param * (1 - self.args.tau) + param * self.args.tau)

            self.num_training += 1
            self.last_map_level = current_map_level
            self.num_training_map_level += 1

    def save(self, name="latest.pt"):
        save_dir = self.paths["checkpoints"]
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "policy": self.policy_net.state_dict(),
            "value": self.value_net.state_dict(),
            "target_value": self.Target_value_net.state_dict(),
            "q1": self.Q_net1.state_dict(),
            "q2": self.Q_net2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "q1_optimizer": self.Q1_optimizer.state_dict(),
            "q2_optimizer": self.Q2_optimizer.state_dict(),
            "num_training": self.num_training,
            "q_scale_ema": self.q_scale_ema,
            "args": vars(self.args),
        }
        save_path = os.path.join(save_dir, name)
        torch.save(ckpt, save_path)
        if name == "latest.pt" and not getattr(self.args, "latest_only_checkpoints", False):
            step_name = "step_{:08d}.pt".format(self.num_training)
            torch.save(ckpt, os.path.join(save_dir, step_name))
        print("====================================")
        print("Models have been saved in {}...".format(save_dir))
        print("====================================")

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.policy_net.load_state_dict(self._maybe_convert_legacy_state_dict(ckpt["policy"], "policy"))
        self.value_net.load_state_dict(self._maybe_convert_legacy_state_dict(ckpt["value"], "value"))
        self.Target_value_net.load_state_dict(
            self._maybe_convert_legacy_state_dict(ckpt["target_value"], "value")
        )
        self.Q_net1.load_state_dict(self._maybe_convert_legacy_state_dict(ckpt["q1"], "q"))
        self.Q_net2.load_state_dict(self._maybe_convert_legacy_state_dict(ckpt["q2"], "q"))
        if "policy_optimizer" in ckpt:
            self._load_optimizer_if_compatible(self.policy_optimizer, ckpt["policy_optimizer"], "policy_optimizer")
        if "value_optimizer" in ckpt:
            self._load_optimizer_if_compatible(self.value_optimizer, ckpt["value_optimizer"], "value_optimizer")
        if "q1_optimizer" in ckpt:
            self._load_optimizer_if_compatible(self.Q1_optimizer, ckpt["q1_optimizer"], "q1_optimizer")
        if "q2_optimizer" in ckpt:
            self._load_optimizer_if_compatible(self.Q2_optimizer, ckpt["q2_optimizer"], "q2_optimizer")
        self.num_training = ckpt.get("num_training", self.num_training)
        self.q_scale_ema = ckpt.get("q_scale_ema", self.q_scale_ema)
        print("====================================")
        print("model has been loaded...")
        print("====================================")

    def _maybe_convert_legacy_state_dict(self, state_dict, module_kind):
        """兼容早期 Prior/CTSAC checkpoint 的权重命名。

        旧版网络把时序模块直接命名为 fc1/transformer/fc2；LiRA-SAC 重构后
        把它们收进 temporal_encoder。这里仅做 key 映射，不改变任何权重数值。
        """
        if any(key.startswith("temporal_encoder.") for key in state_dict):
            return state_dict

        converted = {}
        converted_legacy = False
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("fc1."):
                new_key = "temporal_encoder." + key
                converted_legacy = True
            elif key.startswith("transformer."):
                new_key = "temporal_encoder.temporal." + key
                converted_legacy = True
            elif key.startswith("fc2."):
                new_key = "temporal_encoder." + key
                converted_legacy = True
            converted[new_key] = value

        if converted_legacy:
            current = {
                "policy": self.policy_net,
                "value": self.value_net,
                "q": self.Q_net1,
            }[module_kind].state_dict()
            converted = {key: value for key, value in converted.items() if key in current and current[key].shape == value.shape}
            converted.update({key: value for key, value in current.items() if key not in converted})
            print("[Compat] converted legacy Prior/CTSAC {} checkpoint keys.".format(module_kind))
        return converted

    def _load_optimizer_if_compatible(self, optimizer, optimizer_state, name):
        try:
            optimizer.load_state_dict(optimizer_state)
        except ValueError as exc:
            print("[Compat] skipped incompatible {} state: {}".format(name, exc))
