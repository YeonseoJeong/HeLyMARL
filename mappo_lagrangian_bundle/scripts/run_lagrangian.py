"""Train MAPPO-Lagrangian baseline (joint UE+BS, primal-dual energy constraint).

See docs/additional_lagrangian_marl.md for the algorithm specification.
"""
import argparse
import numpy as np

from lymarl.utils.seed import set_seed
from lymarl.utils.config import load_config, build_sbs_list_with_asym, asym_path
from lymarl.utils.experiment_logger import add_logger_cli_args, build_experiment_logger
from lymarl.wireless.basestation import SmallCellBaseStation
from lymarl.wireless.user_equipment import UserEquipment
from lymarl.wireless.geometry import generate_triangle_coverage
from lymarl.algos.mappo_trainer import MAPPOTrainer
from lymarl.baselines.lagrangian_env import LagrangianEnvironment


def build_env(cfg, use_hard_constraint: bool = False) -> LagrangianEnvironment:
    sc = cfg["scenario"]
    ev = cfg["env"]
    lg = cfg.get("lagrangian", {})
    sbs_pos = generate_triangle_coverage(sc["area_size"], sc["coverage_radius"])
    sbs_list, power_budget_ratio = build_sbs_list_with_asym(cfg, sbs_pos, SmallCellBaseStation)
    users = [
        UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
        for i in range(sc["num_users"])
    ]
    return LagrangianEnvironment(
        base_stations=sbs_list, users=users,
        V=ev["V"], power_budget_ratio=power_budget_ratio,
        enable_mobility=ev["enable_mobility"],
        enable_channel_variation=ev["enable_channel_variation"],
        on_window=ev["on_window"], bs_top_k=ev["bs_top_k"],
        hard_window_len=ev["hard_window_len"],
        bs_over_penalty=ev["bs_over_penalty"],
        eta_q=ev["eta_q"], alpha_rate=ev["alpha_rate"], beta_z=ev["beta_z"],
        use_hard_constraint=use_hard_constraint,
        eta_mu=float(lg.get("eta_mu", 0.5)),
        dual_update_interval=int(lg.get("dual_update_interval", 100)),
        mu_max=float(lg.get("mu_max", 50.0)),
        use_dimensionless=bool(lg.get("use_dimensionless", True)),
        use_ema_penalty=bool(lg.get("use_ema_penalty", True)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true")
    add_logger_cli_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["scenario"]["seed"]
    set_seed(seed)

    tr = cfg["train"]
    n_steps = args.steps if args.steps is not None else tr["n_steps"]
    env = build_env(cfg)
    run_name = f"MAPPO_Lagrangian_seed{seed}"
    logger, log_interval = build_experiment_logger(
        cfg, args, run_name=run_name, tags=["lagrangian"]
    )
    logger.log_config(cfg)

    trainer = MAPPOTrainer(
        env=env,
        lr_actor_ue=tr["lr_actor_ue"], lr_actor_bs=tr["lr_actor_bs"],
        lr_critic_ue=tr["lr_critic_ue"], lr_critic_bs=tr["lr_critic_bs"],
        gamma=tr["gamma"], gae_lambda=tr["gae_lambda"],
        clip_epsilon=tr["clip_epsilon"],
        entropy_coef_ue=tr["entropy_coef_ue"], entropy_coef_bs=tr["entropy_coef_bs"],
        value_coef_ue=tr["value_coef_ue"], value_coef_bs=tr["value_coef_bs"],
        n_epochs=tr["n_epochs"], minibatch_size=tr["minibatch_size"],
    )

    try:
        trainer.train(
            n_steps=n_steps,
            update_interval=tr["update_interval"],
            save_npz_path=asym_path("outputs/logs/lagrangian/MAPPO_Lagrangian_train_rewards.npz", cfg),
            logger=logger,
            log_interval=log_interval,
            run_label="train",
        )
        trainer.save_model(asym_path("outputs/models/MAPPO_Lagrangian.pt", cfg))

        # Why: persist dual-variable trajectory for offline plotting (trainer doesn't know about mu_b).
        if env.mu_b_history:
            np.savez(
                asym_path("outputs/logs/lagrangian/MAPPO_Lagrangian_dual.npz", cfg),
                mu_b_history=np.stack(env.mu_b_history, axis=0),
                C_b_history=np.stack(env.C_b_history, axis=0),
                dual_update_interval=env.dual_update_interval,
                eta_mu=env.eta_mu,
            )

        if not args.no_eval:
            env.set_hard_constraint(True)
            trainer.evaluate(
                n_steps=args.eval_steps,
                save_npz_path=asym_path("outputs/logs/lagrangian/MAPPO_Lagrangian_eval.npz", cfg),
                logger=logger,
                log_interval=log_interval,
                run_label="eval",
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
