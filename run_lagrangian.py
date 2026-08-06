import os
import gc
import numpy as np
import torch

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.trainer_happo import HAPPOTrainer
from baselines.env_constrainedhappo import JensenHAPPOEnvironment, PFHAPPOEnvironment


# ============================================================
# Experiment settings
# ============================================================
TRAIN_SEEDS = [0, 1, 2]
EVAL_SEEDS = [2000, 2001, 2002, 2003, 2004]

VARIANTS = ["pf", "jensen"]
KAPPA_LIST = [0.015]

LAMBDA_E = 0.0

STEPS_PER_EPISODE = 10000
TRAIN_EPISODES = 10
EVAL_EPISODES = 1
UPDATE_INTERVAL = 128

ETA_MU = 0.5
ETA_NU = 0.5
MU_MAX = 100.0
NU_MAX = 100.0

USE_DIMENSIONLESS = False

RUN_TRAIN = True
RUN_EVAL = True

SAVE_DIR = "results/results_baselines"

# ============================================================
# Environment
# ============================================================
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
            (
                np.random.uniform(10, 90),
                np.random.uniform(10, 90),
            ),
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

        # Existing HAPPO constraint settings
        lambda_E=lambda_E,
        kappa=kappa,

        # Constrained-HAPPO dual-variable settings
        eta_mu=eta_mu,
        eta_nu=eta_nu,
        mu_max=mu_max,
        nu_max=nu_max,
        use_dimensionless=use_dimensionless,

        # Dual variables are updated once per episode
        episode_length=hard_window_len,
    )

    if variant == "jensen":
        return JensenHAPPOEnvironment(**common_kwargs)

    if variant == "pf":
        return PFHAPPOEnvironment(**common_kwargs)

    raise ValueError(f"Unknown variant: {variant}")


# ============================================================
# Trainer
# ============================================================
def make_trainer(env, eval_env=None):
    return HAPPOTrainer(
        env=env,
        eval_env=eval_env,
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


# ============================================================
# File paths
# ============================================================
def make_run_dir(variant, kappa, train_seed):
    # .3f is required so that kappa=0.015 is not rounded to 0.01/0.02.
    return os.path.join(
        SAVE_DIR,
        variant,
        f"kappa_{kappa:.3f}_seed_{train_seed}",
    )


def make_model_path(variant, kappa, train_seed):
    return os.path.join(
        make_run_dir(variant, kappa, train_seed),
        "model.pt",
    )


def make_train_npz_path(variant, kappa, train_seed):
    return os.path.join(
        make_run_dir(variant, kappa, train_seed),
        "train.npz",
    )


def make_dual_npz_path(variant, kappa, train_seed):
    return os.path.join(
        make_run_dir(variant, kappa, train_seed),
        "dual_history.npz",
    )


def make_eval_npz_path(variant, kappa, train_seed, eval_seed):
    return os.path.join(
        make_run_dir(variant, kappa, train_seed),
        f"eval_seed_{eval_seed}.npz",
    )


# ============================================================
# Dual-history saving
# ============================================================
def save_dual_history(env, save_path):
    dual_data = {}

    if hasattr(env, "mu_E_b_history") and len(env.mu_E_b_history) > 0:
        dual_data["mu_E_b_history"] = np.stack(
            env.mu_E_b_history,
            axis=0,
        )

    if hasattr(env, "nu_H_u_history") and len(env.nu_H_u_history) > 0:
        dual_data["nu_H_u_history"] = np.stack(
            env.nu_H_u_history,
            axis=0,
        )

    if hasattr(env, "C_E_b_history") and len(env.C_E_b_history) > 0:
        dual_data["C_E_b_history"] = np.stack(
            env.C_E_b_history,
            axis=0,
        )

    if hasattr(env, "C_H_u_history") and len(env.C_H_u_history) > 0:
        dual_data["C_H_u_history"] = np.stack(
            env.C_H_u_history,
            axis=0,
        )

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
    print(f"Saved dual history: {save_path}")


# ============================================================
# Training
# ============================================================
def train_one_model(variant, kappa, train_seed):
    run_dir = make_run_dir(variant, kappa, train_seed)
    os.makedirs(run_dir, exist_ok=True)

    train_npz_path = make_train_npz_path(
        variant,
        kappa,
        train_seed,
    )
    model_path = make_model_path(
        variant,
        kappa,
        train_seed,
    )
    dual_npz_path = make_dual_npz_path(
        variant,
        kappa,
        train_seed,
    )

    print("\n" + "=" * 100)
    print(
        f"TRAIN | Constrained HAPPO-{variant.upper()} | "
        f"kappa={kappa:.3f} | train_seed={train_seed}"
    )
    print("=" * 100)

    env_soft = make_env(
        seed=train_seed,
        variant=variant,
        lambda_E=LAMBDA_E,
        kappa=kappa,
        use_hard_constraint=False,
        hard_window_len=STEPS_PER_EPISODE,
        eta_mu=ETA_MU,
        eta_nu=ETA_NU,
        mu_max=MU_MAX,
        nu_max=NU_MAX,
        use_dimensionless=USE_DIMENSIONLESS,
    )

    # Environment construction consumes RNG states, so reset the seed
    # immediately before network initialization.
    set_seed(train_seed)
    trainer_soft = make_trainer(env_soft)

    trainer_soft.train(
        n_episodes=TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        update_interval=UPDATE_INTERVAL,
        save_npz_path=train_npz_path,
        eval_every=0,
    )

    trainer_soft.save_model(model_path)
    save_dual_history(env_soft, dual_npz_path)

    del trainer_soft
    del env_soft
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Evaluation
# ============================================================
def evaluate_one_model(variant, kappa, train_seed, eval_seed):
    model_path = make_model_path(
        variant,
        kappa,
        train_seed,
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    eval_npz_path = make_eval_npz_path(
        variant,
        kappa,
        train_seed,
        eval_seed,
    )

    print("\n" + "=" * 100)
    print(
        f"EVAL | Constrained HAPPO-{variant.upper()} | "
        f"kappa={kappa:.3f} | "
        f"train_seed={train_seed} | "
        f"eval_seed={eval_seed}"
    )
    print("=" * 100)

    env_hard = make_env(
        seed=eval_seed,
        variant=variant,
        lambda_E=LAMBDA_E,
        kappa=kappa,
        use_hard_constraint=True,
        hard_window_len=STEPS_PER_EPISODE,
        eta_mu=ETA_MU,
        eta_nu=ETA_NU,
        mu_max=MU_MAX,
        nu_max=NU_MAX,
        use_dimensionless=USE_DIMENSIONLESS,
    )

    # Reset before trainer initialization for reproducible evaluation.
    set_seed(eval_seed)
    trainer_hard = make_trainer(env_hard)
    trainer_hard.load_model(model_path)

    # Reset once more immediately before trajectory generation.
    set_seed(eval_seed)
    trainer_hard.evaluate(
        n_episodes=EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        save_npz_path=eval_npz_path,
    )

    del trainer_hard
    del env_hard
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for variant in VARIANTS:
        for kappa in KAPPA_LIST:
            for train_seed in TRAIN_SEEDS:
                if RUN_TRAIN:
                    train_one_model(
                        variant=variant,
                        kappa=kappa,
                        train_seed=train_seed,
                    )

                if RUN_EVAL:
                    for eval_seed in EVAL_SEEDS:
                        evaluate_one_model(
                            variant=variant,
                            kappa=kappa,
                            train_seed=train_seed,
                            eval_seed=eval_seed,
                        )

    print("\n✅ Constrained-HAPPO multi-seed training and evaluation completed!\n")


if __name__ == "__main__":
    main()