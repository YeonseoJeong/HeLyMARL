import numpy as np

from lymarl.wireless.basestation import SmallCellBaseStation
from lymarl.wireless.user_equipment import UserEquipment
from lymarl.wireless.geometry import generate_triangle_coverage
from lymarl.baselines.lagrangian_env import LagrangianEnvironment


def make_env(eta_mu: float = 1.0, dual_update_interval: int = 20):
    positions = generate_triangle_coverage(100, 35)
    sbs = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(positions)]
    users = [UserEquipment(i + 1, (50.0, 50.0)) for i in range(3)]
    return LagrangianEnvironment(
        base_stations=sbs, users=users,
        V=5.0, power_budget_ratio=0.5,
        enable_mobility=False, enable_channel_variation=False,
        on_window=10, bs_top_k=3,
        hard_window_len=100, bs_over_penalty=10.0,
        eta_q=1.0, alpha_rate=1.0, beta_z=1.0,
        use_hard_constraint=False,
        eta_mu=eta_mu,
        dual_update_interval=dual_update_interval,
    )


def _roll(env, bs_on: bool, n_steps: int):
    """Drive env for n_steps with all BSs forced on (UEs request each BS) or off (idle)."""
    # Why: BS activates only when (a) UE requests it and (b) BS picks a valid candidate slot.
    if bs_on:
        ue_actions = {u.ue_id: (i % env.n_bs) for i, u in enumerate(env.users)}
    else:
        ue_actions = {u.ue_id: env.no_request_action for u in env.users}
    bs_choice = 0 if bs_on else env.bs_top_k  # action `bs_top_k` == idle
    for _ in range(n_steps):
        _, _, cand_lists = env.build_bs_decision_inputs(ue_actions)
        bs_actions = {bs.bs_id: bs_choice for bs in env.base_stations}
        env.step_joint(ue_actions, bs_actions, cand_lists)


def test_obs_dims():
    env = make_env()
    local_obs, global_obs = env.reset()
    assert env.local_obs_dim == 2 * env.n_bs
    assert env.bs_obs_dim == 1 + 1 + env.bs_top_k
    assert env.global_obs_dim == env.n_agents * env.n_bs + 2 * env.n_bs
    assert global_obs.shape == (env.global_obs_dim,)
    for obs in local_obs.values():
        assert obs.shape == (env.local_obs_dim,)


def test_mu_b_increases_when_all_bs_on():
    env = make_env(eta_mu=1.0, dual_update_interval=10)
    env.reset()
    mu_before = env.mu_b.copy()
    _roll(env, bs_on=True, n_steps=10)
    # Why: every BS on every slot ⇒ C^b = (1 - rho) * e_bar_b > 0 ⇒ mu_b rises.
    assert np.all(env.mu_b > mu_before), f"mu_b did not increase: {env.mu_b}"


def test_mu_b_projects_to_nonneg_when_all_bs_off():
    env = make_env(eta_mu=1.0, dual_update_interval=10)
    env.reset()
    env.mu_b[:] = 1e-3  # small positive start
    _roll(env, bs_on=False, n_steps=10)
    # Why: C^b = -E_bar_b < 0 ⇒ subgradient drives mu_b below 0, then projection clamps to 0.
    assert np.all(env.mu_b >= 0.0)
    assert np.all(env.mu_b == 0.0)
