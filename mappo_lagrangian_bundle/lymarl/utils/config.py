import os
import yaml
from pathlib import Path
from typing import List, Tuple, Union


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place (override wins on scalar conflicts)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_config(path: str = "configs/default.yaml", override: str = None) -> dict:
    """Load a YAML config file and return it as a nested dict.

    If override is given, deep-merge that YAML on top (override wins on conflicts).
    """
    with open(Path(path), "r") as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(Path(override), "r") as f:
            ov = yaml.safe_load(f) or {}
        _deep_merge(cfg, ov)
    return cfg


def build_sbs_list_with_asym(
    cfg: dict,
    sbs_positions: list,
    SmallCellBaseStationCls,
):
    """Build the small-cell BS list, applying asymmetry knobs when enabled.

    Returns (sbs_list, power_budget_ratio) where power_budget_ratio is either a scalar
    (symmetric) or a per-BS list (asymmetric). Mirrors the wiring already used in
    scripts/train_lymarl.py and scripts/train_unicrit.py so all baseline scripts can
    share the same asym path without duplicating ~10 lines each.
    """
    sc = cfg["scenario"]
    ev = cfg["env"]
    asym = cfg.get("asymmetry", {}) or {}

    if asym.get("enabled", False):
        tx_list = asym["tx_power_dbm_per_bs"]
        ratio_list = asym["power_budget_ratio_per_bs"]
        assert len(tx_list) == len(sbs_positions), \
            "tx_power_dbm_per_bs length must match BS count"
        assert len(ratio_list) == len(sbs_positions), \
            "power_budget_ratio_per_bs length must match BS count"
        sbs_list = [
            SmallCellBaseStationCls(
                i + 1, pos, sc["beam_limit"], sc["coverage_radius"],
                tx_power_dbm=tx_list[i],
            )
            for i, pos in enumerate(sbs_positions)
        ]
        return sbs_list, ratio_list

    sbs_list = [
        SmallCellBaseStationCls(i + 1, pos, sc["beam_limit"], sc["coverage_radius"])
        for i, pos in enumerate(sbs_positions)
    ]
    return sbs_list, ev["power_budget_ratio"]


def asym_path(path: str, cfg: dict) -> str:
    """Route an output path through outputs/<kind>/asym/... when asym is enabled.

    Examples (when cfg.asymmetry.enabled = True):
      outputs/logs/jmarl/JMARL_eval.npz -> outputs/logs/asym/jmarl/JMARL_eval.npz
      outputs/models/JMARL.pt           -> outputs/models/asym/JMARL.pt

    Why: preserves the symmetric artifacts (and downstream figures) when running an
    asymmetric sweep, without forcing every script to grow a --asym flag.
    """
    asym = cfg.get("asymmetry", {}) or {}
    if not asym.get("enabled", False):
        return path
    parts = path.split(os.sep)
    for top in ("logs", "models", "figures", "runs"):
        if top in parts:
            i = parts.index(top)
            parts.insert(i + 1, "asym")
            return os.sep.join(parts)
    # Why: fallback so unconfigured paths don't silently bypass routing.
    return os.path.join("asym", path)
