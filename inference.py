import yaml
import os
import torch
from pytorch_lightning import Trainer
from model import LitSegment
from data import DataWrapper
from augmentations import get_augmentations
from scorer import main as get_score, read_buildings


torch.set_float32_matmul_precision('high')


def run(mode, config_path, output_folder, checkpoint_name='best.ckpt', run_scoring=True, transforms='none') -> None:
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    dm = DataWrapper(config, test=mode)
    lit = LitSegment.load_from_checkpoint(os.path.join(config['training']['save_weights_dir'], config['training']['wandb_run_name'], checkpoint_name))
    dm.train_dataset.transforms = get_augmentations(config['data']['image_h'], config['data']['image_w'], transforms)
    dm.val_dataset.transforms = get_augmentations(config['data']['image_h'], config['data']['image_w'], transforms)
    lit.test_results_folder = output_folder
    lit.test_mode = mode
    lit.conf_prediction = config['prediction']

    trainer = Trainer(
                    accelerator="cuda",
                    devices=[0],
                    )

    trainer.test(lit, dm)

    if run_scoring:
        get_score(output_folder, config['training'][f'{mode}_labels_dir'])
    for name in os.listdir(output_folder):
        try:
            read_buildings(os.path.join(output_folder, name))
        except Exception as e:
            print('heer')


if __name__ == "__main__":
    data = 'test'
    config_path = 'config.yaml'
    checkpoint_name = 'last.ckpt'
    output_folder = f'test_results_{data}'
    run_scoring = False
    transforms = 'none'
    os.makedirs(output_folder, exist_ok=True)


    run(data, config_path, output_folder, checkpoint_name, run_scoring=run_scoring, transforms=transforms)
