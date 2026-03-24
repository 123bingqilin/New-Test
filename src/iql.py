import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import util
from .util import update_exponential_moving_average

EXP_ADV_MAX = 100.0


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


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


def _compute_local_phys_score_and_weight(
    phys_residual,
    w_min=0.05,
    delta=0.0,
    temperature=1.0,
    eps=1e-6,
):
    if phys_residual is None:
        return None, None, None, None

    r = phys_residual.detach().reshape(-1)
    median = torch.median(r)
    mad = torch.median(torch.abs(r - median)).clamp_min(eps)
    z = (r - median) / mad

    temperature = max(float(temperature), float(eps))
    logistic = torch.sigmoid((float(delta) - z) / temperature)
    w = float(w_min) + (1.0 - float(w_min)) * logistic
    w = w.clamp(min=float(w_min), max=1.0)
    return z, median, mad, w


def _compute_global_confidence(
    z,
    prev_global_confidence=1.0,
    confidence_lambda=1.0,
    rho=0.99,
):
    if z is None:
        return None, None, None

    z_pos = torch.clamp(z.detach().reshape(-1), min=0.0)
    confidence_lambda = max(float(confidence_lambda), 0.0)
    rho = min(max(float(rho), 0.0), 1.0)

    psi_t = torch.exp(-confidence_lambda * z_pos).mean()
    global_confidence = rho * float(prev_global_confidence) + (1.0 - rho) * psi_t.item()
    return z_pos, psi_t, global_confidence


def _compute_soft_warmup_alpha(step, t0, t1):
    t = int(step)
    t0 = int(t0)
    t1 = int(t1)

    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    if t < t0:
        return 0.0
    if t < t1:
        return float(t - t0) / float(t1 - t0)
    return 1.0


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
        algo_name="physiql",
        model_mode="separate",
        use_forward=1,
        use_inverse=1,
        use_phys=1,
        lambda_f=1.0,
        lambda_inv=1.0,
        aux_weight=1.0,
        encoder=None,
        forward_model=None,
        inverse_model=None,
        phys_weight_min=0.05,
        phys_delta=0.0,
        phys_tau=1.0,
        phys_mad_eps=1e-6,
        use_global_conf=0,
        phys_global_lambda=1.0,
        phys_global_rho=0.99,
        phys_global_eta=0.5,
        phys_global_init=1.0,
        phys_warmup_start=0,
        phys_warmup_end=0,
    ):
        super().__init__()
        self.algo_name = algo_name
        self.model_mode = model_mode

        self.lambda_f = float(lambda_f)
        self.lambda_inv = float(lambda_inv)
        self.use_forward = bool(use_forward) and (algo_name == "physiql") and (forward_model is not None)
        self.use_inverse = bool(use_inverse) and (algo_name == "physiql") and (inverse_model is not None)
        self.use_phys = bool(use_phys) and (algo_name == "physiql")

        self.qf = qf.to(util.DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(util.DEFAULT_DEVICE)
        self.vf = vf.to(util.DEFAULT_DEVICE)
        self.policy = policy.to(util.DEFAULT_DEVICE)

        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)

        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.aux_weight = aux_weight

        self.phys_weight_min = phys_weight_min
        self.phys_delta = phys_delta
        self.phys_tau = phys_tau
        self.phys_mad_eps = phys_mad_eps

        self.use_global_conf = bool(use_global_conf) and self.use_phys
        self.phys_global_lambda = phys_global_lambda
        self.phys_global_rho = phys_global_rho
        self.phys_global_eta = phys_global_eta
        self.global_confidence = float(phys_global_init)

        self.phys_warmup_start = phys_warmup_start
        self.phys_warmup_end = phys_warmup_end

        self.encoder = None
        self.encoder_optimizer = None
        if encoder is not None and (self.use_forward or self.use_inverse):
            self.encoder = encoder.to(util.DEFAULT_DEVICE)
            self.encoder_optimizer = optimizer_factory(self.encoder.parameters())

        self.forward_model = None
        self.forward_optimizer = None
        if self.use_forward:
            self.forward_model = forward_model.to(util.DEFAULT_DEVICE)
            self.forward_optimizer = optimizer_factory(self.forward_model.parameters())

        self.inverse_model = None
        self.inverse_optimizer = None
        if self.use_inverse:
            self.inverse_model = inverse_model.to(util.DEFAULT_DEVICE)
            self.inverse_optimizer = optimizer_factory(self.inverse_model.parameters())

    def _compute_phys_terms(self, observations, actions, next_observations):
        dyn_residual = None
        inv_residual = None
        forward_loss = torch.tensor(0.0, device=observations.device)
        inverse_loss = torch.tensor(0.0, device=observations.device)

        if not (self.use_forward or self.use_inverse):
            return forward_loss, inverse_loss, dyn_residual, inv_residual

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
        step=0,
        **kwargs,
    ):
        forward_loss = torch.tensor(0.0, device=observations.device)
        inverse_loss = torch.tensor(0.0, device=observations.device)
        dyn_residual = None
        inv_residual = None
        phys_residual = None
        phys_score = None
        phys_score_pos = None
        phys_median = None
        phys_mad = None
        phys_avg_conf = None
        phys_global_conf = None

        ones = torch.ones_like(rewards).reshape(-1)
        w_local_phys = ones
        w_phys = ones
        target_scale = ones
        bootstrap_term = None

        warmup_alpha = (
            _compute_soft_warmup_alpha(
                step=step,
                t0=self.phys_warmup_start,
                t1=self.phys_warmup_end,
            )
            if self.use_phys
            else 0.0
        )

        if self.algo_name == "physiql":
            forward_loss, inverse_loss, dyn_residual, inv_residual = self._compute_phys_terms(
                observations,
                actions,
                next_observations,
            )

            if self.use_phys:
                with torch.no_grad():
                    if dyn_residual is not None and inv_residual is not None:
                        phys_residual = self.lambda_f * dyn_residual + self.lambda_inv * inv_residual
                    elif dyn_residual is not None:
                        phys_residual = self.lambda_f * dyn_residual
                    elif inv_residual is not None:
                        phys_residual = self.lambda_inv * inv_residual

                    if phys_residual is not None:
                        phys_score, phys_median, phys_mad, w_local_phys = _compute_local_phys_score_and_weight(
                            phys_residual,
                            w_min=self.phys_weight_min,
                            delta=self.phys_delta,
                            temperature=self.phys_tau,
                            eps=self.phys_mad_eps,
                        )

                        if self.use_global_conf:
                            phys_score_pos, phys_avg_conf, phys_global_conf = _compute_global_confidence(
                                phys_score,
                                prev_global_confidence=self.global_confidence,
                                confidence_lambda=self.phys_global_lambda,
                                rho=self.phys_global_rho,
                            )
                            self.global_confidence = float(phys_global_conf)
                            w_phys = self.phys_global_eta * float(phys_global_conf) + (1.0 - self.phys_global_eta) * w_local_phys
                            w_phys = w_phys.clamp(min=0.0, max=1.0)
                        else:
                            w_phys = w_local_phys

                        target_scale = (1.0 - warmup_alpha) + warmup_alpha * w_phys

        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)
            bootstrap_term = (1.0 - terminals.float()).reshape(-1) * self.discount * target_scale * next_v.detach().reshape(-1)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        targets = rewards.reshape(-1) + bootstrap_term
        qs = self.qf.both(observations, actions)
        q_loss = sum(torch.mean((q.reshape(-1) - targets) ** 2) for q in qs) / len(qs)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)

        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        policy_loss = torch.mean(exp_adv * bc_losses)

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        if self.algo_name == "physiql":
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad(set_to_none=True)
            if self.forward_optimizer is not None:
                self.forward_optimizer.zero_grad(set_to_none=True)
            if self.inverse_optimizer is not None:
                self.inverse_optimizer.zero_grad(set_to_none=True)

            aux_total = 0.0
            if self.forward_optimizer is not None:
                aux_total = aux_total + forward_loss
            if self.inverse_optimizer is not None:
                aux_total = aux_total + inverse_loss

            if isinstance(aux_total, torch.Tensor) and (self.forward_optimizer is not None or self.inverse_optimizer is not None):
                aux_total.backward()

            if self.encoder_optimizer is not None:
                self.encoder_optimizer.step()
            if self.forward_optimizer is not None:
                self.forward_optimizer.step()
            if self.inverse_optimizer is not None:
                self.inverse_optimizer.step()

        out = {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "adv_mean": adv.mean().item(),
            "exp_adv_mean": exp_adv.mean().item(),
            "weighted_bc_mean": (exp_adv * bc_losses).mean().item(),
        }

        if log_analysis:
            out["phys_warmup_alpha"] = float(warmup_alpha)
            out["target_scale_mean"] = target_scale.mean().item()
            out["target_scale_std"] = target_scale.std().item()
            out["bootstrap_term_mean"] = bootstrap_term.mean().item()
            out["bootstrap_term_std"] = bootstrap_term.std().item()

            _append_quantiles(out, "target_scale", target_scale, [0.05, 0.25, 0.50, 0.75, 0.95])
            _append_group_means(out, "target_scale", target_scale, corrupt_mask)
            _append_quantiles(out, "bootstrap_term", bootstrap_term, [0.05, 0.25, 0.50, 0.75, 0.95])
            _append_group_means(out, "bootstrap_term", bootstrap_term, corrupt_mask)

            if self.use_phys:
                out["w_local_phys_mean"] = w_local_phys.mean().item()
                out["w_local_phys_std"] = w_local_phys.std().item()
                out["w_phys_mean"] = w_phys.mean().item()
                out["w_phys_std"] = w_phys.std().item()

                _append_quantiles(out, "w_local_phys", w_local_phys, [0.05, 0.25, 0.50, 0.75, 0.95])
                _append_group_means(out, "w_local_phys", w_local_phys, corrupt_mask)
                _append_quantiles(out, "w_phys", w_phys, [0.05, 0.25, 0.50, 0.75, 0.95])
                _append_group_means(out, "w_phys", w_phys, corrupt_mask)

            if dyn_residual is not None:
                out["dyn_residual_mean"] = dyn_residual.mean().item()
                _append_quantiles(out, "dyn_residual", dyn_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "dyn_residual", dyn_residual, corrupt_mask)

            if inv_residual is not None:
                out["inv_residual_mean"] = inv_residual.mean().item()
                _append_quantiles(out, "inv_residual", inv_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "inv_residual", inv_residual, corrupt_mask)

            if phys_residual is not None:
                out["phys_residual_mean"] = phys_residual.mean().item()
                _append_quantiles(out, "phys_residual", phys_residual, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "phys_residual", phys_residual, corrupt_mask)

            if phys_score is not None:
                out["phys_score_mean"] = phys_score.mean().item()
                out["phys_score_std"] = phys_score.std().item()
                _append_quantiles(out, "phys_score", phys_score, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "phys_score", phys_score, corrupt_mask)

            if phys_score_pos is not None:
                out["phys_score_pos_mean"] = phys_score_pos.mean().item()
                out["phys_score_pos_std"] = phys_score_pos.std().item()
                _append_quantiles(out, "phys_score_pos", phys_score_pos, [0.50, 0.90, 0.95, 0.99])
                _append_group_means(out, "phys_score_pos", phys_score_pos, corrupt_mask)

            if phys_median is not None:
                out["phys_median"] = phys_median.item()
            if phys_mad is not None:
                out["phys_mad"] = phys_mad.item()
            if phys_avg_conf is not None:
                out["phys_avg_conf"] = phys_avg_conf.item()
            if phys_global_conf is not None:
                out["phys_global_conf"] = float(phys_global_conf)

        return out
