import yaml
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from segmentation.model import LitSegment
from data import DataWrapper

torch.set_float32_matmul_precision('high')

def run(config_path) -> None:
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = DataWrapper(config)
    lit = LitSegment(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        dirpath=os.path.join(config['training']['save_weights_dir'], config['training']['wandb_run_name']),
        filename="model-{epoch:02d}",
        save_top_k=1,
        save_last=True,
        mode="max",
    )

    os.makedirs(os.path.join(config['training']['save_logs_dir'], config['training']['wandb_run_name']), exist_ok=True)
    logger = WandbLogger(project=config['training']["wandb_project_name"], name=config['training']["wandb_run_name"], save_dir=os.path.join(config['training']['save_logs_dir'], config['training']['wandb_run_name']))

    trainer = Trainer(logger=logger,
                      accelerator="gpu",
                      devices=[0],
                      max_epochs=config['training']["epochs"],
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=0,
                      )

    trainer.fit(lit, dm)
    print(f'Loading best checkpoint: {checkpoint_callback.best_model_path}')
    best_model = LitSegment.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, dm)

if __name__ == "__main__":
    config_path = 'segmentation/config_seg.yaml'
    run(config_path)
