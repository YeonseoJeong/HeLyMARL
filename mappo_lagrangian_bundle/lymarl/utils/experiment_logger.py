import argparse
import os
from typing import Dict, Iterable, List, Optional


class NullExperimentLogger:
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        return

    def log_config(self, config: Dict):
        return

    def close(self):
        return


class TensorBoardExperimentLogger:
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise ImportError(
                "TensorBoard logging requires 'tensorboard'. Install with: pip install -e .[logging]"
            ) from exc
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            tag = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(tag, float(v), int(step))

    def log_config(self, config: Dict):
        text = str(config)
        self.writer.add_text("config", text, global_step=0)

    def close(self):
        self.writer.flush()
        self.writer.close()


class WandbExperimentLogger:
    def __init__(
        self,
        project: str,
        run_name: str,
        entity: Optional[str] = None,
        mode: str = "online",
        config: Optional[Dict] = None,
        tags: Optional[Iterable[str]] = None,
    ):
        try:
            import wandb
        except Exception as exc:
            raise ImportError(
                "W&B logging requires 'wandb'. Install with: pip install -e .[logging]"
            ) from exc
        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            mode=mode,
            config=config if config is not None else {},
            tags=list(tags) if tags is not None else None,
        )

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        payload = {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            key = f"{prefix}/{k}" if prefix else k
            payload[key] = float(v)
        if payload:
            self._wandb.log(payload, step=int(step))

    def log_config(self, config: Dict):
        if self.run is not None:
            self.run.config.update(config, allow_val_change=True)

    def close(self):
        if self.run is not None:
            self.run.finish()


class CompositeExperimentLogger:
    def __init__(self, loggers: List):
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step, prefix=prefix)

    def log_config(self, config: Dict):
        for logger in self.loggers:
            logger.log_config(config)

    def close(self):
        for logger in self.loggers:
            logger.close()


def _get_arg(args: Optional[argparse.Namespace], key: str):
    if args is None:
        return None
    return getattr(args, key, None)


def add_logger_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--logger", choices=["none", "tensorboard", "wandb", "both"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)


def build_experiment_logger(cfg: Dict, args: Optional[argparse.Namespace], run_name: str, tags=None):
    lc = cfg.get("logging", {}) if isinstance(cfg, dict) else {}

    enabled_cfg = bool(lc.get("enabled", False))
    backend_cfg = str(lc.get("backend", "tensorboard"))
    logger_arg = _get_arg(args, "logger")

    if logger_arg is None:
        if not enabled_cfg:
            return NullExperimentLogger(), int(lc.get("log_interval", 100))
        backend = backend_cfg
    else:
        if logger_arg == "none":
            return NullExperimentLogger(), int(lc.get("log_interval", 100))
        backend = logger_arg

    resolved_run_name = _get_arg(args, "run_name") or lc.get("run_name") or run_name
    resolved_log_dir = _get_arg(args, "log_dir") or lc.get("log_dir", "outputs/runs")
    resolved_project = _get_arg(args, "wandb_project") or lc.get("project", "LyMARL")
    resolved_entity = _get_arg(args, "wandb_entity") or lc.get("entity")
    resolved_mode = _get_arg(args, "wandb_mode") or lc.get("mode", "online")
    log_interval = int(lc.get("log_interval", 100))

    loggers = []
    if backend in ("tensorboard", "both"):
        tb_dir = os.path.join(resolved_log_dir, resolved_run_name)
        loggers.append(TensorBoardExperimentLogger(tb_dir))
    if backend in ("wandb", "both"):
        loggers.append(
            WandbExperimentLogger(
                project=resolved_project,
                run_name=resolved_run_name,
                entity=resolved_entity,
                mode=resolved_mode,
                config=cfg,
                tags=tags,
            )
        )

    if not loggers:
        return NullExperimentLogger(), log_interval
    if len(loggers) == 1:
        return loggers[0], log_interval
    return CompositeExperimentLogger(loggers), log_interval
