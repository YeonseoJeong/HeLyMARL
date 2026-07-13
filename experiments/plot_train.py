import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.2,
})

def moving_avg(x, window=100):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x

    out = np.zeros_like(x, dtype=np.float32)
    csum = 0.0
    for i in range(len(x)):
        csum += float(x[i])
        if i >= window:
            csum -= float(x[i - window])
            out[i] = csum / window
        else:
            out[i] = csum / (i + 1)
    return out


def plot_train_reward_loss(
    npz_path,
    save_dir=None,
    reward_smooth_window=1000,
    title_prefix="HeLyMARL",
    steps_per_episode=10000,
    compare_reward_npz_paths=None,
):
    data = np.load(npz_path)

    if save_dir is None:
        save_dir = os.path.dirname(npz_path) if os.path.dirname(npz_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(npz_path))[0]

    # ====================================================== 
    # 1) Reward plot 
    # ====================================================== 
    def _load_reward(npz_file): 
        d = np.load(npz_file) 

        if ( "reward_x_1000" in d and "global_reward_1000" in d and len(d["global_reward_1000"]) > 0 ): 
            x = np.asarray(d["reward_x_1000"], dtype=np.float32) 
            y = np.asarray(d["global_reward_1000"], dtype=np.float32) 
            label_suffix = "block avg 1000" 

        elif "global_reward" in d and len(d["global_reward"]) > 0: 
            raw = np.asarray(d["global_reward"], dtype=np.float32) 
            y = moving_avg(raw, reward_smooth_window) 
            x = np.arange(1, len(y) + 1) 
            label_suffix = f"MA {reward_smooth_window}" 

        else: 
            x, y, label_suffix = None, None, None 

        d.close() 
        return x, y, label_suffix
    
    # ------------------------------------------------------
    # Case A: compare multiple algorithms
    # Normalized reward comparison
    # compare_reward_npz_paths = {
    #     "PF-HAPPO": "...npz",
    #     "Jensen-HAPPO": "...npz",
    #     "HeLyMARL": "...npz",
    # }
    # ------------------------------------------------------
    if compare_reward_npz_paths is not None:

        def _normalize(y):
            y = np.asarray(y, dtype=np.float32)
            if len(y) == 0:
                return y

            y_min = float(np.min(y))
            y_max = float(np.max(y))

            if y_max - y_min < 1e-8:
                return np.zeros_like(y, dtype=np.float32)

            return (y - y_min) / (y_max - y_min)

        plt.figure(figsize=(8, 5))

        plotted = False
        for alg_name, alg_npz_path in compare_reward_npz_paths.items():
            if not os.path.exists(alg_npz_path):
                print(f"[Warning] File not found: {alg_npz_path}")
                continue

            reward_x, reward_y, label_suffix = _load_reward(alg_npz_path)

            if reward_x is None:
                print(f"[Warning] No reward data found: {alg_npz_path}")
                continue

            reward_y_norm = _normalize(reward_y)

            plt.plot(
                reward_x,
                reward_y_norm,
                linewidth=2.0,
                label=f"{alg_name} ({label_suffix}, normalized)"
            )
            plotted = True

        if plotted:
            plt.xlabel("Training step")
            plt.ylabel("Normalized reward")
            plt.title(f"{title_prefix} Training Reward Comparison")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            reward_path = os.path.join(save_dir, f"{base}_reward_comparison_normalized.png")
            plt.savefig(reward_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved normalized reward comparison plot: {reward_path}")
        else:
            plt.close()
            print("[Warning] No reward data found for comparison.")

    # ------------------------------------------------------
    # Case B: original single algorithm reward plot
    # ------------------------------------------------------
    else:
        reward_x, reward_y, label_suffix = _load_reward(npz_path)

        if reward_x is not None:
            episode_x = reward_x / steps_per_episode

            plt.figure(figsize=(7, 4.5))

            # --------------------------------------------------
            # 1) 기존 이동평균 또는 block-average reward
            # --------------------------------------------------
            plt.plot(
                episode_x,
                reward_y,
                linewidth=2.0,
                color="#1F4E79" ,
                label=f"Global reward ({label_suffix})"
            )
            # --------------------------------------------------
            # 2) Episode별 평균 reward 계산
            # --------------------------------------------------
            with np.load(npz_path) as d:
                if "global_reward" in d and len(d["global_reward"]) > 0:
                    raw_reward = np.asarray(d["global_reward"], dtype=np.float32).reshape(-1)
                    n_episodes = len(raw_reward) // steps_per_episode
                    if n_episodes > 0:
                        trimmed_reward = raw_reward[:n_episodes * steps_per_episode]
                        episode_reward_matrix = trimmed_reward.reshape(n_episodes, steps_per_episode)
                        episode_mean_reward = np.mean(episode_reward_matrix, axis=1)
                        episode_idx = np.arange(1, n_episodes + 1)
                        # episode_idx = np.arange(n_episodes, dtype=np.float32) + 0.5
                        plt.plot(
                            episode_idx,
                            episode_mean_reward,
                            marker='o',
                            linestyle='--',
                            linewidth=1.5,
                            color="#D55E00",
                            markersize=4,
                            label="Episode-average reward"
                        )

                        # Episode별 평균값 출력
                        print("\nEpisode-average reward")
                        print("=" * 50)

                        for ep, mean_reward in zip(
                            episode_idx,
                            episode_mean_reward
                        ):
                            print(
                                f"Episode {ep:2d}: "
                                f"{mean_reward:.6f}"
                            )
                        
                    
            max_episode = int(np.ceil(np.max(episode_x)))
            
            for ep in range(1, max_episode + 1):
                plt.axvline(x=ep, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

            plt.xlabel("Training Episode")
            plt.ylabel("Reward")
            # plt.title(f"{title_prefix} Training Reward")
            plt.grid(True, alpha=0.3)
            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.legend(loc= "lower right")
            plt.tight_layout()

            reward_path = os.path.join(save_dir, f"{base}_reward.png")
            plt.savefig(reward_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved reward plot: {reward_path}")
        else:
            print("[Warning] No reward data found.")
    # ======================================================
    # 2) Critic loss plot
    # ======================================================
    update_steps = data["update_steps"] if "update_steps" in data else None
    max_step = 50000

    if "critic_loss" in data and len(data["critic_loss"]) > 0:
        critic_loss = np.asarray(data["critic_loss"], dtype=np.float32)

        if update_steps is not None and len(update_steps) == len(critic_loss):
            x = update_steps
        else:
            x = np.arange(1, len(critic_loss) + 1)

        mask = x < max_step
        x = x[mask]
        critic_loss = critic_loss[mask]

        critic_loss_smooth = moving_avg(critic_loss, window=5)

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, critic_loss, linewidth=1.2, alpha=0.35, label="Critic loss")
        plt.plot(x, critic_loss_smooth, linewidth=2.0, label="Critic loss (MA 5)")
        plt.xlabel("Training step" if update_steps is not None else "Update index")
        plt.ylabel("Critic loss")
        plt.title(f"{title_prefix} Critic Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        critic_loss_path = os.path.join(save_dir, f"{base}_critic_loss.png")
        plt.savefig(critic_loss_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved critic loss plot: {critic_loss_path}")
    else:
        print("[Warning] No critic loss data found.")

    # ======================================================
    # 3) Actor loss plot
    # ======================================================
    actor_keys = {
        "actor_ue_loss": "UE actor loss",
        "actor_bs_loss": "BS actor loss",
    }

    available_actor_losses = {
        k: data[k]
        for k in actor_keys
        if k in data and len(data[k]) > 0
    }

    if len(available_actor_losses) > 0:
        plt.figure(figsize=(7, 4.5))

        for key, label in actor_keys.items():
            if key not in available_actor_losses:
                continue

            actor_loss = np.asarray(available_actor_losses[key], dtype=np.float32)

            if update_steps is not None and len(update_steps) == len(actor_loss):
                x = update_steps
            else:
                x = np.arange(1, len(actor_loss) + 1)

            actor_loss_smooth = moving_avg(actor_loss, window=100)

            plt.plot(x, actor_loss_smooth, linewidth=2.0, label=f"{label} (MA 100)")

        plt.xlabel("Training step" if update_steps is not None else "Update index")
        plt.ylabel("Actor loss")
        plt.title(f"{title_prefix} Actor Losses")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        actor_loss_path = os.path.join(save_dir, f"{base}_actor_loss.png")
        plt.savefig(actor_loss_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved actor loss plot: {actor_loss_path}")
    else:
        print("[Warning] No actor loss data found.")

    # ======================================================
    # 3) Entropy plot
    # ======================================================
    entropy_keys = {
        "entropy_ue": "UE entropy loss",
        "entropy_bs": "BS entropy loss",
    }

    available_entropy = {
        k: data[k]
        for k in entropy_keys
        if k in data and len(data[k]) > 0
    }

    if len(available_entropy) > 0:
        plt.figure(figsize=(7, 4.5))

        for key, label in entropy_keys.items():
            if key not in available_entropy:
                continue

            y = np.asarray(available_entropy[key], dtype=np.float32)

            if update_steps is not None and len(update_steps) == len(y):
                x = update_steps
            else:
                x = np.arange(1, len(y) + 1)

            plt.plot(x, y, linewidth=2.0, label=label)

        plt.xlabel("Training step" if update_steps is not None else "Update index")
        plt.ylabel("Entropy loss")
        plt.title(f"{title_prefix} Entropy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        entropy_path = os.path.join(save_dir, f"{base}_entropy.png")
        plt.savefig(entropy_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved entropy plot: {entropy_path}")
    else:
        print("[Warning] No entropy data found.")


if __name__ == "__main__":
    plot_train_reward_loss(
        npz_path="results/results_kappa/HeLyMARL_train_rewards_kappa_0.03.npz",
        save_dir="results/results_kappa/plots",
        title_prefix="HeLyMARL kappa=0.03",
        steps_per_episode=10000,
        compare_reward_npz_paths={
            "PF-HAPPO": (
                "results/results_baselines/"
                "ConstrainedHAPPO_pf_train_rewards_kappa_0.03.npz"
            ),
            "Jensen-HAPPO": (
                "results/results_baselines/"
                "ConstrainedHAPPO_jensen_train_rewards_kappa_0.03.npz"
            ),
            "HeLyMARL": (
                "results/results_kappa/"
                "HeLyMARL_train_rewards_kappa_0.03.npz"
            ),
        }
    )