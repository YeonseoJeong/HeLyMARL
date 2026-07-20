import os
import numpy as np

from env.basestation import SmallCellBaseStation
from env.user_equipment import UserEquipment
from env.core import generate_triangle_coverage

from HeLyMARL.utils_happo import set_seed
from HeLyMARL.env_happo import HAPPOEnvironment
from HeLyMARL.trainer_happo import HAPPOTrainer


# ============================================================
# Environment
# ============================================================
def make_env(seed, 
             V, 
             lambda_E, 
             kappa, 
             use_hard_constraint, 
             hard_window_len=10000):
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
        V=V,
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

# ============================================================
# Trainer
# ============================================================
def make_trainer(env, eval_env = None):
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
        minibatch_size=256
    )

if __name__ == "__main__":
    train_seeds = [0, 1, 2]
    # checkpoint_eval_seeds = [1000,1001,1002,]
    final_eval_seed = 2000
    v = 5.0
    kappa_list = [0.01, 0.02, 0.03]
    lambda_E = 0.0

    steps_per_episode = 10000
    train_episodes = 10
    eval_episode = 1
    update_interval = 128

    save_dir = "results/results_kappa"
    os.makedirs(save_dir, exist_ok=True)

    for kappa in kappa_list:
        kappa_tag = f"{kappa:.2f}"

        # for seed in train_seeds:
            # print("\n" + "=" * 100)
            # print(
            #     f"Training HeLyMARL | "
            #     f"kappa={kappa_tag} | "
            #     f"seed={seed}"
            # )

            # env_soft = make_env(
            #     seed=seed,
            #     V=v,
            #     lambda_E=lambda_E,
            #     kappa=kappa,
            #     use_hard_constraint=False,
            #     hard_window_len=steps_per_episode,
            # )

            # set_seed(seed)

            # env_checkpoint_eval = make_env(
            #     seed=checkpoint_eval_seeds[0],
            #     V=v,
            #     lambda_E=lambda_E,
            #     kappa=kappa,
            #     use_hard_constraint=True,
            #     hard_window_len=steps_per_episode,
            # )
            
            # trainer_soft = make_trainer(env_soft, eval_env = None)

        train_npz_path = os.path.join(
            save_dir,
            (
                f"HeLyMARL_train_rewards_"
                f"kappa_{kappa_tag}.npz"
                # f"seed_{seed}.npz"
            ),
        )

        model_path = os.path.join(
            save_dir,
            (
                f"HeLyMARL_model_"
                f"kappa_{kappa_tag}.pt"
                # f"seed_{seed}.pt"
            ),
        )

            # trainer_soft.train(
            #     n_episodes=train_episodes,
            #     steps_per_episode=steps_per_episode,
            #     update_interval=update_interval,
            #     save_npz_path=train_npz_path,
            #     eval_every=0,
            #     eval_n_episodes=len(checkpoint_eval_seeds),
            #     eval_steps_per_episode=steps_per_episode,
            #     eval_seeds=checkpoint_eval_seeds,
            #     eval_deterministic=False,
            #     policy_improvement_dir=f"{save_dir}/checkpoints",
            #     checkpoint_every_updates_early=8,
            #     checkpoint_every_updates_mid=40,
            #     checkpoint_every_updates_late=80,
            #     checkpoint_early_until_step=10000,
            #     checkpoint_mid_until_step=50000,
            #     save_episode_end_checkpoint=True,
            # )

            # trainer_soft.train(
            #     n_episodes=train_episodes,
            #     steps_per_episode=steps_per_episode,
            #     update_interval=update_interval,
            #     save_npz_path=train_npz_path,

            #     # Episode 중간 evaluation 사용 안 함
            #     eval_every=0,

            #     # Policy-improvement checkpoint 저장 및 평가 안 함
            #     policy_improvement_dir=None,
            #     save_episode_end_checkpoint=False,
            # )
            
            # trainer_soft.save_model(model_path)
        
        print("\n" + "=" * 100)
        print(
            f"Hard evaluation | "
            f"kappa={kappa_tag}"
        )
        print("=" * 100)


        env_hard = make_env(seed=final_eval_seed, V = v, lambda_E = lambda_E, kappa=kappa, use_hard_constraint=True, hard_window_len=steps_per_episode)
        trainer_hard = make_trainer(env_hard)
        trainer_hard.load_model(model_path)
        set_seed(final_eval_seed)
        
        hard_eval_npz_path = os.path.join(
            save_dir,
            (
                f"HeLyMARL_eval_hard_"
                f"kappa_{kappa_tag}.npz"
            ),
        )
        
        trainer_hard.evaluate(
            n_episodes=eval_episode,
            steps_per_episode=steps_per_episode,
            save_npz_path=hard_eval_npz_path
        )

    print("\n✅ Completed!\n")