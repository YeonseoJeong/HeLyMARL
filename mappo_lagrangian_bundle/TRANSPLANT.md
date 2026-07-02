# MAPPO-Lagrangian — Transplant Guide

**What this is:** A MAPPO baseline with per-BS primal-dual energy constraints. Each BS has its own Lagrange multiplier `mu_b` that is updated via projected sub-gradient ascent. The reward penalty adapts over time instead of using a fixed energy coefficient.

---

## Package layout

```
lymarl/
  baselines/
    jmarl_env.py        ← J-MARL (parent): joint UE+BS MAPPO, no Lyapunov queues
    lagrangian_env.py   ← MAPPO-Lagrangian: adds per-BS mu_b, dual update
  env/
    mappo_env.py        ← base heterogeneous two-population MAPPO environment
  algos/
    networks.py         ← UEActorNetwork, BSActorNetwork, centralized critics
    value_norm.py       ← running mean/std value normalization
    mappo_trainer.py    ← standard PPO loop (GAE, minibatches, two-team)
  wireless/
    basestation.py      ← SmallCellBaseStation / MacroBaseStation
    user_equipment.py   ← UserEquipment
    geometry.py         ← generate_triangle_coverage
  utils/
    seed.py, config.py, stats.py, experiment_logger.py
scripts/
  run_lagrangian.py     ← end-to-end training entry point
configs/
  lagrangian_smoke.yaml ← hyperparameters (smoke-test sized)
tests/
  test_lagrangian_env.py
```

---

## Dependency chain

```
LagrangianEnvironment
  └─ JMARLEnvironment       (jmarl_env.py)
       └─ MAPPOEnvironment  (env/mappo_env.py)
            └─ SmallCellBaseStation, UserEquipment, geometry
MAPPOTrainer                (algos/mappo_trainer.py)
  └─ UEActorNetwork, BSActorNetwork, CentralizedCriticUE/BS  (algos/networks.py)
  └─ ValueNorm / ValueNormVec                                 (algos/value_norm.py)
```

All imports are within the `lymarl` package — no hidden external dependencies beyond `torch`, `numpy`, `matplotlib`, `pyyaml`.

---

## How to transplant

### Option A — install as a package (recommended)

```bash
# drop the lymarl/ folder into your repo, then:
pip install -e .          # uses the bundled pyproject.toml
```

Then import normally:
```python
from lymarl.baselines.lagrangian_env import LagrangianEnvironment
from lymarl.algos.mappo_trainer import MAPPOTrainer
```

### Option B — copy as a flat module

If you don't want a separate package, copy `lymarl/` directly into your repo root.  
Adjust any absolute imports (`from lymarl.xxx import ...`) to match your directory structure.

---

## Minimal run

```bash
python -m scripts.run_lagrangian --config configs/lagrangian_smoke.yaml --steps 5000 --no-eval
```

Artifacts written to `outputs/logs/lagrangian/` and `outputs/models/`.

---

## Key hyperparameters (`configs/lagrangian_smoke.yaml → lagrangian:`)

| param | default | meaning |
|---|---|---|
| `eta_mu` | `1e-3` | dual step size (sub-gradient ascent on `mu_b`) |
| `dual_update_interval` | `1000` | steps between `mu_b` updates |
| `mu_max` | `50.0` (code default) | projection upper bound on `mu_b` |
| `use_dimensionless` | `True` | constraint is `on_ratio_b - rho` (no `e_bar_b` scaling) |
| `use_ema_penalty` | `True` | per-step penalty uses EMA on-ratio instead of instantaneous |

Set `use_dimensionless=True` (the default) unless you are reproducing the raw-energy formulation — `e_bar_b ~ 0.1 W` otherwise squashes the constraint signal by ~10×.

---

## Observation dimensions (after `LagrangianEnvironment.__init__`)

| obs | formula | what's added vs J-MARL |
|---|---|---|
| UE local | `2 * n_bs` | unchanged |
| BS local | `1 + 1 + bs_top_k` | `+1` for `mu_b[bi]` |
| Global | `n_agents*n_bs + 2*n_bs` | `+n_bs` for `mu_b` vector |

If you add or change observation components, update both `mappo_env.py` and `networks.py` to keep dims in sync.

---

## Dual-variable trajectory

After training, `env.mu_b_history` (list of `(n_bs,)` arrays) and `env.C_b_history` (constraint residuals) are available for plotting. `run_lagrangian.py` saves them to `outputs/logs/lagrangian/MAPPO_Lagrangian_dual.npz`.

---

## Tests

```bash
pytest tests/test_lagrangian_env.py -v
```

Three smoke tests: observation dimensions, `mu_b` rises when all BSs are forced on, `mu_b` stays non-negative when all BSs are idle.
