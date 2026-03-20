import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, update_exponential_moving_average


EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


def _append_quantiles(out, name, x, quantiles):
    if x is None:
        return

    x = x.detach().reshape(-1)
    if x.numel() == 0:
        return

    q = torch.tensor(quantiles, device=x.device, dtype=x.dtype)
    vals = torch.quantile(x, q)
    for qq, vv in zip(quantiles, vals):
        out[f"{name}_q{int(qq * 100):02d}"] = vv.item()


def _append_group_means(out, name, x, corrupt_mask):
    if x is None or corrupt_mask is None:
        return

    x = x.detach().reshape(-1)
    mask = corrupt_mask.detach().reshape(-1) > 0.5

    if mask.numel() != x.numel():
        return

    if mask.any():
        out[f"{name}_corrupt_mean"] = x[mask].mean().item()
    if (~mask).any():
        out[f"{name}_clean_mean"] = x[~mask].mean().item()


class ImplicitQLearning(nn.Module):
    def __init__(
        self,
        qf,
        vf,
        policy,
        optimizer_factory,
        max_steps,
        tau,
        beta,
        discount=0.99,
        alpha=0.005,
        lambda_inv=1.0,
        alpha_phys=1.0,
        aux_weight=1.0,
        algo_name="physiql",
        model_mode="separate",
        use_forward=1,
        use_inverse=1,
        use_phys=1,
        encoder=None,
        forward_model=None,
        inverse_model=None,
    ):
        super().__init__()

        self.algo_name = algo_name
        self.model_mode = model_mode
        self.use_forward = bool(use_forward)
        self.use_inverse = bool(use_inverse)
        self.use_phys = bool(use_phys)

        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)

        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

        self.lambda_inv = lambda_inv
        self.alpha_phys = alpha_phys
        self.aux_weight = aux_weight

        self.encoder = None
        self.encoder_optimizer = None
        if encoder is not None:
            self.encoder = encoder.to(DEFAULT_DEVICE)
            self.encoder_optimizer = optimizer_factory(self.encoder.parameters())

        self.forward_model = None
        self.forward_optimizer = None
        if forward_model is not None and self.use_forward:
            self.forward_model = forward_model.to(DEFAULT_DEVICE)
            self.forward_optimizer = optimizer_factory(self.forward_model.parameters())

        self.inverse_model = None
        self.inverse_optimizer = None
        if inverse_model is not None and self.use_inverse:
            self.inverse_model = inverse_model.to(DEFAULT_DEVICE)
            self.inverse_optimizer = optimizer_factory(self.inverse_model.parameters())

    def _compute_phys_terms(self, observations, actions, next_observations):
        dyn_residual = None
        inv_residual = None

        forward_loss = torch.tensor(0.0, device=observations.device)
        inverse_loss = torch.tensor(0.0, device=observations.device)

        if self.model_mode == "shared":
            feat = self.encoder(observations)
            next_feat = self.encoder(next_observations)
        else:
            feat = observations
            next_feat = next_observations

        if self.use_forward and self.forward_model is not None:
            if self.model_mode == "shared":
                pred_next_obs = self.forward_model(feat, actions)
            else:
                pred_next_obs = self.forward_model(observations, actions)

            dyn_residual = torch.mean((pred_next_obs - next_observations) ** 2, dim=1)
            forward_loss = self.aux_weight * dyn_residual.mean()

        if self.use_inverse and self.inverse_model is not None:
            if self.model_mode == "shared":
                pred_actions = self.inverse_model(feat, next_feat)
            else:
                pred_actions = self.inverse_model(observations, next_observations)

            inv_residual = torch.mean((pred_actions - actions) ** 2, dim=1)
            inverse_loss = self.aux_weight * inv_residual.mean()

        return forward_loss, inverse_loss, dyn_residual, inv_residual

    def update(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        corrupt_mask=None,
        log_analysis=False,
        **kwargs,
    ):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # value
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # q
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # defaults
        forward_loss = torch.tensor(0.0, device=observations.device)
        inverse_loss = torch.tensor(0.0, device=observations.device)
        dyn_residual = None
        inv_residual = None
        phys_residual = None

        w_phys = torch.ones_like(adv).squeeze(-1) if adv.ndim > 1 else torch.ones_like(adv)

        # auxiliary modules only for physiql
        if self.algo_name == "physiql":
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad(set_to_none=True)

            forward_loss, inverse_loss, dyn_residual, inv_residual = self._compute_phys_terms(
                observations,
                actions,
                next_observations,
            )

            if self.use_forward and self.forward_optimizer is not None:
                self.forward_optimizer.zero_grad(set_to_none=True)
            if self.use_inverse and self.inverse_optimizer is not None:
                self.inverse_optimizer.zero_grad(set_to_none=True)

            aux_total = 0.0
            if self.use_forward and self.forward_model is not None:
                aux_total = aux_total + forward_loss
            if self.use_inverse and self.inverse_model is not None:
                aux_total = aux_total + inverse_loss

            if isinstance(aux_total, torch.Tensor):
                aux_total.backward()

            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
            if self.use_forward and self.forward_optimizer is not None:
                self.forward_optimizer.step()
            if self.use_inverse and self.inverse_optimizer is not None:
                self.inverse_optimizer.step()

            with torch.no_grad():
                if dyn_residual is not None and inv_residual is not None:
                    phys_residual = dyn_residual + self.lambda_inv * inv_residual
                elif dyn_residual is not None:
                    phys_residual = dyn_residual
                elif inv_residual is not None:
                    phys_residual = self.lambda_inv * inv_residual

                if self.use_phys and phys_residual is not None:
                    w_phys = torch.exp(-self.alpha_phys * phys_residual).clamp(min=1e-3, max=1.0)
                else:
                    w_phys = torch.ones_like(adv).squeeze(-1) if adv.ndim > 1 else torch.ones_like(adv)

        # policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        policy_loss = torch.mean(w_phys * exp_adv * bc_losses)

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        out = {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "adv_mean": adv.mean().item(),
            "exp_adv_mean": exp_adv.mean().item(),
            "weighted_bc_mean": (w_phys * exp_adv * bc_losses).mean().item(),
        }

        if log_analysis:
            _append_quantiles(out, "w_phys", w_phys, [0.05, 0.25, 0.50, 0.75, 0.95])
            _append_group_means(out, "w_phys", w_phys, corrupt_mask)

            if dyn_residual is not None:
                _append_quantiles(out, "dyn_residual", dyn_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "dyn_residual", dyn_residual, corrupt_mask)

            if inv_residual is not None:
                _append_quantiles(out, "inv_residual", inv_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "inv_residual", inv_residual, corrupt_mask)

            if phys_residual is not None:
                _append_quantiles(out, "phys_residual", phys_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "phys_residual", phys_residual, corrupt_mask)

            if self.algo_name == "physiql":
                if self.use_forward:
                    out["forward_loss"] = forward_loss.item()
                if self.use_inverse:
                    out["inverse_loss"] = inverse_loss.item()
                if dyn_residual is not None:
                    out["dyn_residual_mean"] = dyn_residual.mean().item()
                if inv_residual is not None:
                    out["inv_residual_mean"] = inv_residual.mean().item()
                if phys_residual is not None:
                    out["phys_residual_mean"] = phys_residual.mean().item()
                out["w_phys_mean"] = w_phys.mean().item()
                out["w_phys_std"] = w_phys.std().item()

        return out