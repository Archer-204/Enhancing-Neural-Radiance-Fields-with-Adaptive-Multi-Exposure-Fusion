import data
import framework
import os
import shutil
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Callback, LightningDataModule, LightningModule
import torch
import yaml
from torchvision.utils import save_image

class SaveTestImagesCallback(Callback):
    def __init__(self, src_dir):
        super().__init__()
        self.src_dir = src_dir
        self.dest_dir = self.src_dir.replace('LOM_full', 'LOM_full_enhanced')
        self.save_dir = os.path.join(self.dest_dir, 'low')
        self.files_to_copy = [
            'transforms_test.json',
            'transforms_val.json',
            'transforms_train.json',
            'high'
        ]
        self.create_directory()

    def create_directory(self):
        os.makedirs(self.dest_dir, exist_ok=True)
        for item in self.files_to_copy:
            src_path = os.path.join(self.src_dir, item)
            dest_path = os.path.join(self.dest_dir, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                print(f"Copied directory {src_path} to {dest_path}")
            elif os.path.isfile(src_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"Copied file {src_path} to {dest_path}")
            else:
                print(f"{src_path} does not exist and cannot be copied.")


    def on_test_start(self, trainer, pl_module):
        os.makedirs(self.save_dir, exist_ok=True)
        trainer.logger = None
        print(f"Test images will be saved to: {self.save_dir}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            print(f"No outputs in batch {batch_idx}, skipping.")
            return
        generated_images = outputs['pred_image']
        paths = outputs['path']
        for img, path in zip(generated_images, paths):
            save_path = os.path.join(self.save_dir, os.path.basename(path))
            img = img.detach().cpu()
            save_image(img, save_path)
            print(f"Saved {save_path}")

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pipeline", choices=["full", "train", "test"])
        parser.add_argument("--exp_name")
        parser.add_argument("--version")
        parser.add_argument("--checkpoint")
        parser.add_argument("--dirpath", default='./source/ckpt', help="Directory where to save the checkpoints")
        parser.add_argument("--test_image_dir", default='./test_images', help="Directory where to save the test images")

    def before_fit(self):
        os.makedirs(self.config["dirpath"], exist_ok=True)
        print(f"before_fit called, ensuring dirpath exists: {self.config['dirpath']}")
        print("Callbacks in trainer:")
        for callback in self.trainer.callbacks:
            print(callback)
        for ckpt in self.trainer.callbacks:
            if isinstance(ckpt, ModelCheckpoint):
                print(f"Updating ModelCheckpoint dirpath to {self.config['dirpath']}")
                ckpt.dirpath = self.config["dirpath"]

if __name__ == "__main__":
    with open('./source/configs/lom.yaml', 'r') as file:
        config = yaml.safe_load(file)
        # print(config)
    cli = CustomLightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    filename='last-checkpoint-{epoch:02d}',
                    save_top_k=1,
                    mode='min',
                    save_last=True,
                    every_n_epochs=1,
                )
            ],
            "logger": False,
            "default_root_dir": "/tmp"
        },
    )

    if cli.config["pipeline"] == "full":
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])
        cli.trainer.test(cli.model, cli.datamodule)
    elif cli.config["pipeline"] == "train":
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])
    elif cli.config["pipeline"] == "test":
        save_test_images_callback = SaveTestImagesCallback(
            src_dir=config['data']['init_args']['data_root']
        )
        cli.trainer.callbacks.append(save_test_images_callback)
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])