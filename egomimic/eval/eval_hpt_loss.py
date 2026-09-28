"""HPT validation loss and action sampling without video rendering."""

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment


class HPTLossEval(Eval):
    def __init__(self):
        self.trainer = None
        self.model = None

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.model.forward_eval(batch)
        losses = self.model.compute_losses(predictions, batch)
        batch_size = sum(
            sample[self.model.ac_keys[embodiment]].shape[0]
            for embodiment, sample in batch.items()
        )
        self.trainer.lightning_module.log(
            "Valid/action_loss",
            losses["action_loss"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        for embodiment, sample in batch.items():
            key = f"{get_embodiment(embodiment).lower()}_loss"
            self.trainer.lightning_module.log(
                f"Valid/{key}",
                losses[key],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=sample[self.model.ac_keys[embodiment]].shape[0],
            )
