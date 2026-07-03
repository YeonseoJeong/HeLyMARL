import os
import numpy as np
import matplotlib.pyplot as plt


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
):
    data = np.load(npz_path)

    if save_dir is None:
        save_dir = os.path.dirname(npz_path) if os.path.dirname(npz_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(npz_path))[0]

    # ======================================================
    # 1) Reward plot
    # ======================================================
    if "reward_x_1000" in data and "global_reward_1000" in data and len(data["global_reward_1000"]) > 0:
        reward_x = data["reward_x_1000"]
        reward_y = data["global_reward_1000"]
        reward_label = "Global reward (block avg 1000)"
    elif "global_reward" in data and len(data["global_reward"]) > 0:
        reward_raw = data["global_reward"]
        reward_y = moving_avg(reward_raw, reward_smooth_window)
        reward_x = np.arange(1, len(reward_y) + 1)
        reward_label = f"Global reward (MA {reward_smooth_window})"
    else:
        reward_x = None
        reward_y = None

    if reward_x is not None:
        plt.figure(figsize=(7, 4.5))
        plt.plot(reward_x, reward_y, linewidth=2.0, label=reward_label)
        plt.xlabel("Training step")
        plt.ylabel("Reward")
        plt.title(f"{title_prefix} Training Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
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
    max_step = 10000

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
        title_prefix="HeLyMARL kappa=0.03"
    )