from lightning import Callback


class LogDataModuleCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        pl_module.logger.log_hyperparams(
            {
                "datamodule_name": datamodule.__class__.__name__,
                "batch_size": datamodule.batch_size,
                "dataset_size": len(datamodule.train_dataset),
            }
        )
