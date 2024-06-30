import glob
import os
import cv2
import torch
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from torch.utils import data
from torch.utils.data import DataLoader


def is_image_file(file_path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

class PairedDataset(data.Dataset):
    def __init__(self, root_folder, pattern, get_label_fn, resize=None, return_name=False):
        super().__init__()
        path = os.path.join(root_folder, pattern)
        all_files = sorted(glob.glob(path, recursive=True))
        self.data_list = [f for f in all_files if is_image_file(f)]
        self.gt_list = [get_label_fn(p) for p in self.data_list]
        self.resize = resize
        self.return_name = return_name

    def read_image(self, path):
        im = cv2.imread(path)
        assert im is not None, path
        im = im[:, :, ::-1]
        if self.resize is not None:
            im = cv2.resize(im, (self.resize, self.resize))
        im = im / 255.0
        im = torch.from_numpy(im).float().permute(2, 0, 1)
        return im

    def __getitem__(self, index):
        input_path = self.data_list[index]
        input_im = self.read_image(input_path)
        gt_im = self.read_image(self.gt_list[index])
        if self.return_name:
            return input_im, gt_im, os.path.join(*input_path.split("/")[-2:])
        return (
            input_im,
            gt_im,
        )

    def __len__(self):
        return len(self.data_list)


class LOMDataModule(LightningDataModule):
    def __init__(self, data_root, num_workers):
        super().__init__()

        def get_label_fn(path):
            return path.replace("low", "high")

        test_data = PairedDataset(data_root, "./low/*.*", get_label_fn, resize=None, return_name=True)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return self.test_loader

