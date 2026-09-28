"""Save an explicit end-of-training checkpoint independently of best selection."""

from pathlib import Path

from lightning.pytorch import Callback


class SaveFinalCheckpoint(Callback):
    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(
            str(Path(trainer.default_root_dir) / "checkpoints" / "final.ckpt")
        )
