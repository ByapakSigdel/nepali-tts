"""
Tiny Lightning callback that writes live training progress to a plain text file every N steps,
so the progress tracker can show real epoch/step numbers (Lightning's progress bar doesn't write
to a redirected, non-terminal log).

Wired in via configs/train_ne.yaml (class_path: status_writer.StatusWriter); the training script
puts this folder on PYTHONPATH so it's importable.
"""
import time

from lightning.pytorch.callbacks import Callback


class StatusWriter(Callback):
    def __init__(
        self,
        path: str = "/home/byapak/voicemodel/models/ne_stageA/status.txt",
        every_n_steps: int = 10,
    ):
        super().__init__()
        self.path = path
        self.every_n_steps = max(1, int(every_n_steps))

    def _loss(self, trainer):
        try:
            for k in ("loss_gen_all", "loss_g", "g_total", "total_loss", "loss", "train_loss"):
                v = trainer.callback_metrics.get(k)
                if v is not None:
                    return f"{float(v):.3f}"
        except Exception:
            pass
        return ""

    def _write(self, trainer, batch_idx):
        try:
            spe = int(trainer.num_training_batches)
        except Exception:
            spe = 0
        try:
            with open(self.path, "w") as f:
                f.write(
                    f"epoch={trainer.current_epoch} "
                    f"global_step={trainer.global_step} "
                    f"step_in_epoch={batch_idx + 1}/{spe} "
                    f"loss={self._loss(trainer)} "
                    f"ts={int(time.time())}\n"
                )
        except Exception:
            pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx % self.every_n_steps) == 0:
            self._write(trainer, batch_idx)
