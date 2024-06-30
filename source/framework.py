import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class ImageEvaluator(nn.Module):
    def __init__(self, ps=25, exposed_level=0.5):
        super(ImageEvaluator, self).__init__()
        self.ps = ps
        self.exposed_level = exposed_level
        self.pad = nn.ReflectionPad2d(ps // 2)
        self.pool = nn.AvgPool2d(ps, stride=1)

    def forward(self, images):
        eps = 1 / 255.0
        max_rgb, _ = torch.max(images, dim=1, keepdim=True)
        min_rgb, _ = torch.min(images, dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb + eps) / (max_rgb + eps)
        mean_rgb = self.calculate_mean_rgb(images)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps
        contrast = self.calculate_contrast(images, mean_rgb)
        score = self.calculate_iqa_score(saturation, contrast, exposedness)
        return score

    def calculate_mean_rgb(self, images):
        padded_images = self.pad(images)
        mean_rgb = self.pool(padded_images).mean(dim=1, keepdim=True)
        return mean_rgb

    def calculate_contrast(self, images, mean_rgb):
        padded_images = self.pad(images)
        squared_images = padded_images * padded_images
        mean_squared = self.pool(squared_images).mean(dim=1, keepdim=True)
        contrast = mean_squared - mean_rgb ** 2
        return contrast

    def calculate_iqa_score(self, saturation, contrast, exposedness):
        score = torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
        return score



def save_image(image, path):
    base_dir = os.path.split(path)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(image, path)


class TVLoss(torch.nn.Module):
    def forward(self, x):
        x = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        return torch.mean(h_tv) + torch.mean(w_tv)


class ExposureSimulator:
    def __init__(self, model, gamma_low, gamma_high, num_refs, evaluator):
        self.model = model
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.num_refs = num_refs
        self.evaluator = evaluator

    def __call__(self, image):
        return self.simulate_exposure(image)

    def simulate_exposure(self, image):
        bs, ch, h, w = image.shape
        underexposed_gamma = self.generate_gamma(image, self.gamma_high, positive=True)
        overexposed_gamma = self.generate_gamma(image, self.gamma_low, positive=False)
        gammas = torch.cat([underexposed_gamma, overexposed_gamma], dim=1)

        synthetic_references = self.create_synthetic_references(image, gammas)
        previous_iter_output = self.model(image)[0].clone().detach()
        references = self.combine_references(image, previous_iter_output, synthetic_references)

        scores = self.evaluate_references(references)
        mul_exp = self.select_best_reference(references, scores, bs, ch)

        return mul_exp.squeeze(1)

    def generate_gamma(self, image, gamma_limit, positive=True):
        bs = image.shape[0]
        step_size = gamma_limit / self.num_refs if positive else -gamma_limit / self.num_refs
        ranges = torch.linspace(0 if positive else gamma_limit, gamma_limit if positive else 0,
                                steps=self.num_refs + 1).to(image.device)[:-1]
        rand_vals = torch.rand([bs, self.num_refs], device=image.device)
        gamma = torch.exp(rand_vals * step_size + ranges[None, :])
        return gamma

    def create_synthetic_references(self, image, gammas):
        return 1 - (1 - image[:, None]) ** gammas[:, :, None, None, None]

    def combine_references(self, image, previous_iter_output, synthetic_references):
        return torch.cat([image[:, None], previous_iter_output[:, None], synthetic_references], dim=1)

    def evaluate_references(self, references):
        bs, nref, ch, h, w = references.shape
        scores = self.evaluator(references.view(bs * nref, ch, h, w))
        return scores.view(bs, nref, 1, h, w)

    def select_best_reference(self, references, scores, bs, ch):
        max_idx = torch.argmax(scores, dim=1)
        max_idx = max_idx.repeat(1, ch, 1, 1)[:, None]
        return torch.gather(references, 1, max_idx)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor):
        super(ConvBlock, self).__init__()
        expansion_channels = int(out_channels * expansion_factor)  # Ensure this is an integer
        padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(in_channels, expansion_channels, 1, 1, 0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(expansion_channels, expansion_channels, kernel_size, stride, padding, groups=expansion_channels, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(expansion_channels, out_channels, 1, 1, 0, bias=True)
        ]
        self.conv = nn.Sequential(*layers)
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        output = self.conv(x)
        if self.use_res_connect:
            output = x + output
        return output


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()
        base_number = 16
        expansion_factor = 1.5

        self.first_conv = ConvBlock(3, 3, 3, 1, 2)
        self.conv1 = ConvBlock(3, base_number, 3, 2, expansion_factor)
        self.conv2 = ConvBlock(base_number, base_number, 3, 1, expansion_factor)
        self.conv3 = ConvBlock(base_number, base_number * 2, 3, 2, 3)
        self.conv4 = ConvBlock(base_number * 2, base_number * 2, 3, 1, 3)
        self.conv5 = ConvBlock(base_number * 2, base_number, 3, 1, 3)
        self.conv6 = ConvBlock(base_number * 2, base_number, 3, 1, 3)
        self.conv7 = ConvBlock(base_number, 3, 3, 1, expansion_factor)
        self.last_conv = ConvBlock(6, 3, 3, 1, 3)

    def forward(self, x):
        original_input = x
        x = self.first_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        intermediate_feature = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.interpolate(x, (intermediate_feature.shape[2], intermediate_feature.shape[3]), mode='bilinear',
                          align_corners=True)
        x = self.conv6(torch.cat([intermediate_feature, x], dim=1))
        x = self.conv7(x)
        x = F.interpolate(x, (original_input.shape[2], original_input.shape[3]), mode='bilinear', align_corners=True)
        x = self.last_conv(torch.cat([self.first_conv(original_input), x], dim=1))
        x = torch.abs(x + 1)
        img = 1 - (1 - original_input) ** x
        return img, x

class AMENet(LightningModule):
    def __init__(self, tv_weight, gamma_low, gamma_high, num_refs, learning_rate):
        super().__init__()
        self.tv_weight = tv_weight
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.num_refs = num_refs
        self.model = RefinementNet()
        self.eva = ImageEvaluator()
        self.learning_rate = learning_rate
        self.exposure_simulator = ExposureSimulator(self.model, gamma_low, gamma_high, num_refs, self.eva)
        self.mse_loss = torch.nn.MSELoss()
        self.tv_loss = TVLoss()
        self.saved_input = None
        self.saved_mul_exp = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=[0.9, 0.99])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}

    def rotate_if_needed(self, image, target_shape):
        if image.shape[2:] != target_shape:
            image = torch.rot90(image, k=1, dims=[2, 3])
        return image

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["total_loss"])

    def exposure_simulator(self, image):
        return self.exposure_simulator.simulate_exposure(image)

    def validation_step(self, batch, batch_idx):
        pred_image, pred_gamma = self.model(batch)
        self.logger.experiment.add_images("val_input", batch, self.current_epoch)
        self.logger.experiment.add_images("val_output", pred_image, self.current_epoch)

        input_image = batch
        pseudo_gt = self.exposure_simulator(input_image)

        mse_loss = self.mse_loss(pred_image, pseudo_gt)
        tv_loss = self.tv_loss(pred_gamma)
        val_loss = mse_loss + self.tv_weight * tv_loss

        self.log("val_loss", val_loss, on_epoch=True, on_step=False)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx, test_idx=0):
        input_image, path = batch[0], batch[-1]
        pred_image, pred_gamma = self.model(input_image)

        if len(batch) == 3:
            gt_image = batch[1]
            pred_image = self.rotate_if_needed(pred_image, gt_image.shape[2:])

        return {'pred_image': pred_image, 'path': path}

