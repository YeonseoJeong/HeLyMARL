import os
import numpy as np

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.trainer_happo import HAPPOTrainer
from baselines.env_constrainedhappo import JensenHAPPOEnvironment, PFHAPPOEnvironment

def make_env(
    seed,
    variant,
    lambda_E,
    kappa,
    use_hard_constraint,
    hard_window_len=10000,
    eta_mu=0.5,
    eta_nu=0.5,
    mu_max=100.0,
    nu_max=100.0,
    use_dimensionless=True,
    pf_avg_beta=0.99,
):
    set_seed(seed)

    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)

    sbs_list = [
        SmallCellBaseStation(i + 1, pos, 10, 35)
        for i, pos in enumerate(sbs_positions)
    ]

    users = [
        UserEquipment(
            i + 1,
            (np.random.uniform(10, 90), np.random.uniform(10, 90)),
        )
        for i in range(num_users)
    ]

    common_kwargs = dict(
        base_stations=sbs_list,
        users=users,
        V=5.0,
        power_budget_ratio=0.6,
        enable_mobility=True,
        enable_channel_variation=True,
        on_window=100,
        bs_top_k=5,
        hard_window_len=hard_window_len,
        bs_over_penalty=100.0,
        use_hard_constraint=use_hard_constraint,

        # 기존 HAPPO constraint 관련
        lambda_E=lambda_E,
        kappa=kappa,

        # Constrained HAPPO dual variable 관련
        eta_mu=eta_mu,
        eta_nu=eta_nu,
        mu_max=mu_max,
        nu_max=nu_max,
        use_dimensionless=use_dimensionless,
        pf_avg_beta=pf_avg_beta,

        # dual update를 episode 단위로 하기 위한 길이
        episode_length=hard_window_len,
    )

    if variant == "jensen":
        return JensenHAPPOEnvironment(**common_kwargs)

    if variant == "pf":
        return PFHAPPOEnvironment(**common_kwargs)

    raise ValueError(f"Unknown variant: {variant}")


def make_trainer(env):
    return HAPPOTrainer(
        env=env,
        lr_actor_ue=3e-4,
        lr_actor_bs=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef_ue=0.05,
        entropy_coef_bs=0.05,
        value_coef=0.5,
        n_epochs=4,
        minibatch_size=256,
    )

def save_dual_history(env, save_path):
    dual_data = {}

    if hasattr(env, "mu_E_b_history") and len(env.mu_E_b_history) > 0:
        dual_data["mu_E_b_history"] = np.stack(env.mu_E_b_history, axis=0)
    
    if hasattr(env, "nu_H_u_history") and len(env.nu_H_u_history) > 0:
        dual_data["nu_H_u_history"] = np.stack(env.nu_H_u_history, axis=0)
    
    if hasattr(env, "C_E_b_history") and len(env.C_E_b_history) > 0:
        dual_data["C_E_b_history"] = np.stack(env.C_E_b_history, axis=0)
    
    if hasattr(env, "C_H_u_history") and len(env.C_H_u_history) > 0:
        dual_data["C_H_u_history"] = np.stack(env.C_H_u_history, axis=0)

    dual_data["eta_mu"] = env.eta_mu
    dual_data["eta_nu"] = env.eta_nu
    dual_data["mu_max"] = env.mu_max
    dual_data["nu_max"] = env.nu_max
    dual_data["episode_idx"] = env.episode_idx
    dual_data["use_dimensionless"] = env.use_dimensionless
    dual_data["episode_length"] = env.episode_length
    dual_data["power_budget_ratio"] = env.power_budget_ratio
    dual_data["kappa"] = env.kappa

    np.savez(save_path, **dual_data)


if __name__ == "__main__":
    seed = 0

    variants = ["jensen", "pf"]
    kappa_list = [0.03]
    lambda_E = 15.0

    steps_per_episode = 10000
    train_episodes = 5
    eval_episode = 1
    update_interval = 128

    eta_mu = 0.5
    eta_nu = 0.5
    mu_max = 100.0
    nu_max = 100.0

    save_dir = "results/results_baselines"
    os.makedirs(save_dir, exist_ok=True)

    for variant in variants:
        for kappa in kappa_list:
            print(f"\n=== Training Constrained HAPPO-{variant.upper()} | kappa = {kappa} ===")

            env_soft = make_env(
                seed=seed,
                variant=variant,
                lambda_E=lambda_E,
                kappa=kappa,
                use_hard_constraint=False,
                hard_window_len=steps_per_episode,
                eta_mu=eta_mu,
                eta_nu=eta_nu,
                mu_max=mu_max,
                nu_max=nu_max,
                use_dimensionless=True,
                pf_avg_beta=0.9,
            )

            trainer_soft = make_trainer(env_soft)

            train_npz_path = (
                f"{save_dir}/ConstrainedHAPPO_{variant}_train_rewards_kappa_{kappa}.npz"
            )

            model_path = (
                f"{save_dir}/ConstrainedHAPPO_{variant}_model_kappa_{kappa}.pt"
            )

            dual_npz_path = (
                f"{save_dir}/ConstrainedHAPPO_{variant}_dual_history_kappa_{kappa}.npz"
            )

            trainer_soft.train(
                n_episodes=train_episodes,
                steps_per_episode=steps_per_episode,
                update_interval=update_interval,
                save_npz_path=train_npz_path,
            )

            trainer_soft.save_model(model_path)

            save_dual_history(env_soft, dual_npz_path)

            print(f"\n=== Hard Eval Constrained HAPPO-{variant.upper()} | kappa = {kappa} ===")

            env_hard = make_env(
                seed=seed,
                variant=variant,
                lambda_E=lambda_E,
                kappa=kappa,
                use_hard_constraint=True,
                hard_window_len=steps_per_episode,
                eta_mu=eta_mu,
                eta_nu=eta_nu,
                mu_max=mu_max,
                nu_max=nu_max,
                use_dimensionless=True,
                pf_avg_beta=0.9,
            )

            trainer_hard = make_trainer(env_hard)
            trainer_hard.load_model(model_path)

            hard_eval_npz_path = (
                f"{save_dir}/ConstrainedHAPPO_{variant}_eval_hard_kappa_{kappa}.npz"
            )

            trainer_hard.evaluate(
                n_episodes=eval_episode,
                steps_per_episode=steps_per_episode,
                save_npz_path=hard_eval_npz_path,
            )

    print("\n✅ Completed!\n")