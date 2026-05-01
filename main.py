import os
import numpy as np

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage, generate_five_bs_coverage

from lymarl_extension.utils_happo import set_seed
from lymarl_extension.env_happo import HAPPOEnvironment
from lymarl_extension.trainer_happo import HAPPOTrainer


def make_env(seed, lambda_E, kappa, use_hard_constraint, hard_window_len=10000):
    set_seed(seed)

    area_size = 100
    num_users = 20

    sbs_positions = generate_triangle_coverage(area_size, 35)
    sbs_list = [SmallCellBaseStation(i + 1, pos, 10, 35) for i, pos in enumerate(sbs_positions)]
    
    users = [
        UserEquipment(i + 1, (np.random.uniform(10, 90), np.random.uniform(10, 90)))
        for i in range(num_users)
    ]

    env = HAPPOEnvironment(
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
        use_hard_constraint=use_hard_constraint,   # training: no hard constraint
        lambda_E=lambda_E,
        kappa=kappa
    )
    return env

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
        minibatch_size=256
    )

if __name__ == "__main__":
    seed = 42
    kappa_list = [0.01, 0.02, 0.03]
    lambda_E = 15.0

    os.makedirs("results_kappa", exist_ok=True)

    for kappa in kappa_list:
        print(f"\n=== Training with kappa = {kappa} ===")

        # --------------------------------------------------
        # Soft Training
        # --------------------------------------------------
        env_soft = make_env(seed, lambda_E, kappa=kappa, use_hard_constraint=False)
        trainer_soft = make_trainer(env_soft)

        train_steps = 30000
        train_npz_path = f"results_kappa/LyMARL_train_rewards_kappa_{kappa}.npz"
        model_path = f"results_kappa/LyMARL_model_kappa_{kappa}.pt"

        trainer_soft.train(
            n_steps=train_steps,
            update_interval=128,
            save_npz_path=train_npz_path
        )

        trainer_soft.save_model(model_path)
    
        print(f"\n=== Eval only with kappa = {kappa} ===")
        model_path = f"results_kappa/LyMARL_model_kappa_{kappa}.pt"
        if not os.path.exists(model_path):
            print(f"[Error] Model not found: {model_path}")
            continue

        # --------------------------------------------------
        # hard constraint OFF eval
        # --------------------------------------------------
        # trainer_soft.load_model(model_path)
        # soft_eval_npz_path = f"results_kappa/LyMARL_eval_soft_kappa_{kappa}.npz"
        # trainer_soft.evaluate(
        #     n_steps=30000,
        #     save_npz_path=soft_eval_npz_path
        # )

        # --------------------------------------------------
        # hard constraint ON eval
        # --------------------------------------------------
        env_hard = make_env(seed, lambda_E, kappa=kappa, use_hard_constraint=True)
        trainer_hard = make_trainer(env_hard)
        trainer_hard.load_model(model_path)
        
        hard_eval_npz_path = f"results_kappa/LyMARL_eval_hard_kappa_{kappa}.npz"
        trainer_hard.evaluate(
            n_steps=30000,
            save_npz_path=hard_eval_npz_path
        )

    print("\n✅ Completed!\n")