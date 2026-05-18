from lightning.pytorch.callbacks import Callback


class BestMetricTracker(Callback):
    def __init__(self, monitor: str, mode: str = "min"):
        assert mode in {"min", "max"}
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_better(self, current, best):
        if self.mode == "min":
            return current < best
        return current > best

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if self.best is None or self._is_better(current, self.best):
            self.best = current.item()
            pl_module.log(f"best_{self.monitor}", self.best, sync_dist=True)
