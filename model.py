import skimage.measure
import torch
import os
import numpy as np
import skimage
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
# from segmentation.model import LitSegment as Segmentator
from segmentation.sam import LitSegment as Segmentator
from utility import mask2labelme
from scorer import get_score


class LitSegment(pl.LightningModule):
    def __init__(self, config):
        super(LitSegment, self).__init__()
        self.save_hyperparameters()

        self.config = self.hparams['config']
        self.conf_model_height = self.config['model_height']
        self.conf_training = self.config['training']
        self.conf_prediction = self.config['prediction']
        self.conf_data = self.config['data']

        torch.hub.set_dir(self.conf_training['cache_dir'])
        
        self.model_roof = Segmentator.load_from_checkpoint(self.conf_model_height['model_roof_checkpoint'])
        self.model_roof.freeze()

        in_channels = self.conf_model_height["channels"] + 1
        self.model_height = smp.create_model(self.conf_model_height["model_name"], self.conf_model_height["encoder_name"], encoder_depth=self.conf_model_height["encoder_depth"], encoder_weights=self.conf_model_height["encoder_weights"], in_channels=in_channels, classes=1)
        self.test_results_folder = self.conf_training['save_results_dir']

    def check_if_augs_present(self, mode):
        return self.conf_data[f'{mode}_augs'] in ['easy', 'medium', 'hard']

    def forward_roof(self, x):
        y_hat_roof = self.model_roof.forward_roof(x)
        return y_hat_roof
    
    def forward_height(self, x):
        # if self.conf_model_height["encoder_weights"] is not None:
        #     x = (x - self.mean_height) / self.std_height
        mask = self.model_height(x)
        return mask
    
    def loss(self, pred, gt):
        loss = nn.functional.mse_loss(pred, gt, reduction='mean')
        return loss

    def form_input_to_height_model(self, x, mask):
        inputt = torch.cat((x, mask), dim=1)
        return inputt
    
    def correct_target(self, y_hat_roof, mask_height):
        temp_y_hat_roof = y_hat_roof.squeeze(1).cpu().numpy()
        temp_mask_height = mask_height.squeeze(1).cpu()

        y_hat_roof_np = np.uint8((temp_y_hat_roof > 0.5))

        results = []
        for y_hat_roof_element, mask_height_element in zip(y_hat_roof_np, temp_mask_height):
            target = torch.zeros(y_hat_roof_element.shape)
            connected_regions = skimage.measure.label(y_hat_roof_element)
            max_label = connected_regions.max()
            connected_regions = torch.from_numpy(connected_regions)
            for i in range(1, max_label + 1):
                area_of_interest = mask_height_element.where(connected_regions == i, -1).flatten()
                area_of_interest = area_of_interest[area_of_interest > 0]
                if area_of_interest.nelement() != 0:
                    target[connected_regions == i] = area_of_interest[area_of_interest >= 0].mode()[0]
                else:
                    target[connected_regions == i] = 0

            results.append(target)

        return torch.stack(results, dim=0).unsqueeze(1).cuda()

    def get_target_for_height(self, y_hat_roof, mask_height):
        if self.conf_training['target_mode'] == 'tc':
            return self.correct_target(y_hat_roof, mask_height)
        else:
            return mask_height


    def get_seg_mask_for_height(self, mode, mask_roof, pred_mask_roof):
        if mode == 'val':
            return pred_mask_roof
        else:
            if self.conf_training['target_mode'] == 'aps':
                return mask_roof.float()
            else:
                return pred_mask_roof

    def resize_if_needed(self, tensor):
        if tensor.shape[-1] != self.conf_data['original_image_size']:
            tensor = F.interpolate(tensor, size=(self.conf_data['original_image_size'], )*2, mode='bilinear', align_corners=False)
        return tensor
    
    def shared_step(self, batch, mode):
        x = batch["image"]
        mask_roof = batch["mask_roof"]
        mask_height = batch["mask_height"]
        y_hat_roof = self.forward_roof(x)
        y_hat_roof_detached = y_hat_roof.detach()
        y_hat_height = self.forward_height(self.form_input_to_height_model(x, self.get_seg_mask_for_height(mode, mask_roof, y_hat_roof_detached)))
        loss = self.loss(y_hat_height, self.get_target_for_height(y_hat_roof, mask_height))

        # y_hat_roof = (y_hat_roof > 0.5).float()


        mask_roof = (self.resize_if_needed(mask_roof.float()) > 0.5).long()
        mask_height = self.resize_if_needed(mask_height)
        y_hat_roof = self.resize_if_needed(y_hat_roof)
        y_hat_height = self.resize_if_needed(y_hat_height)
        y_hat_roof = (y_hat_roof > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(y_hat_roof, mask_roof, mode='binary', threshold=0.5)
        precision_score = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        recall_score = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")


        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{mode}_precision_builtin", precision_score, on_step=False, on_epoch=True)
        self.log(f"{mode}_recall_builtin", recall_score, on_step=False, on_epoch=True)
        self.log(f"{mode}_iou_builtin", iou_score, on_step=False, on_epoch=True)

        self.predict_batch(batch['name'], y_hat_roof, y_hat_height, self.conf_training[f'save_temp_{mode}_results_dir'], self.conf_prediction)
        if self.check_if_augs_present(mode):
            self.label_batch(batch['name'], mask_roof, mask_height, mode)

        return {"loss": loss, "name": batch["name"], "image": x, "pred_mask_roof": y_hat_roof, "mask_roof": mask_roof.float(), "pred_mask_height": y_hat_height, "mask_height": mask_height}

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, "train")
        return results['loss']

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, "val")
        return results['loss']
    
    def on_epoch_start(self, mode):
        os.makedirs(self.conf_training[f'save_temp_{mode}_results_dir'], exist_ok=True)
        os.makedirs(self.conf_training[f'{mode}_labels_dir_gt'], exist_ok=True)

        if self.check_if_augs_present(mode):
            self.conf_training[f'{mode}_gt_compare_dir'] = self.conf_training[f'{mode}_labels_dir_gt']
        else:
            self.conf_training[f'{mode}_gt_compare_dir'] = self.conf_training[f'{mode}_labels_dir']

    def on_train_epoch_start(self):
        self.on_epoch_start('train')

    def on_validation_epoch_start(self):
        self.on_epoch_start('val')

    def on_test_epoch_start(self):
        os.makedirs(self.conf_training['save_results_dir'], exist_ok=True)

    def on_epoch_end(self, mode):
        try:
            score_obj = get_score(self.conf_training[f'save_temp_{mode}_results_dir'], self.conf_training[f'{mode}_gt_compare_dir'])
            precision, recall, rmse, score = score_obj.get_score(print_detailed_score=False)
        except Exception as e:
            print(e)
            precision, recall, rmse, score = (0, 0, 0, 0)
        self.log(f"{mode}_score", score)
        self.log(f"{mode}_precision", precision)
        self.log(f"{mode}_recall", recall)
        self.log(f"{mode}_rmse", rmse)

    def on_train_epoch_end(self):
        self.on_epoch_end('train')

    def on_validation_epoch_end(self):
        self.on_epoch_end('val')
    
    def test_step(self, batch):
        x = batch['image']
        names = batch['name']

        if x.dim() == 5:
            y_hat_roof_list = []
            y_hat_height_list = []
            for i in range(x.shape[1]):
                x_i = x[:, i, ...]
                y_hat_roof_i = self.forward_roof(x_i)
                y_hat_height_i = self.forward_height(self.form_input_to_height_model(x_i, y_hat_roof_i))
                y_hat_roof_list.append(y_hat_roof_i)
                y_hat_height_list.append(y_hat_height_i)
            y_hat_roof = torch.stack(y_hat_roof_list, dim=1).mean(dim=1)
            y_hat_height = torch.stack(y_hat_height_list, dim=1).mean(dim=1)
        else:
            y_hat_roof = self.forward_roof(x)
            y_hat_height = self.forward_height(self.form_input_to_height_model(x, y_hat_roof.detach()))

        y_hat_roof = self.resize_if_needed(y_hat_roof)
        y_hat_height = self.resize_if_needed(y_hat_height)
        y_hat_roof = (y_hat_roof > 0.5).float()

        self.predict_batch(names, y_hat_roof, y_hat_height, self.test_results_folder, self.conf_prediction)

        if 'mask_roof' in batch and 'mask_height' in batch:
            self.label_batch(names, batch['mask_roof'], batch['mask_height'], self.test_mode)

    def form_predicion(self, name, y_roof, y_height, output_folder, conf_prediction):
        pred_mask_roof = y_roof.squeeze(0).cpu().detach().numpy()
        pred_mask_height = y_height.squeeze(0).cpu().detach().numpy() 
        mask2labelme(pred_mask_roof, pred_mask_height, os.path.join(output_folder, f"{name}.json"), **conf_prediction)

    def predict(self, element, output_folder, conf_prediction):
        x = element['image'].cuda()
        name = element['name']
        y_roof = self.forward_roof(x)
        y_hat_height = self.forward_height(self.form_input_to_height_model(x, y_roof.detach()))

        y_hat_roof = self.resize_if_needed(y_hat_roof).squeeze(0)
        y_hat_height = self.resize_if_needed(y_hat_height).squeeze(0)
        y_hat_roof = (y_hat_roof > 0.5).float()
        
        self.form_predicion(name, y_hat_roof, y_hat_height, output_folder, conf_prediction)
        
    def predict_batch(self, names, y_roofs, y_heights, output_dir, conf_prediction):
        for name, y_roof, y_height in zip(names, y_roofs, y_heights):
            self.form_predicion(name, y_roof, y_height, output_dir, conf_prediction)

    def label_batch(self, names, masks_roof, masks_height, mode='train'):
        for name, m_roof, m_height in zip(names, masks_roof, masks_height):
            m_roof = m_roof.squeeze(0).cpu().detach().numpy()
            m_height = m_height.squeeze(0).cpu().detach().numpy() 
            mask2labelme(m_roof, m_height, os.path.join(self.conf_training[f'{mode}_labels_dir_gt'], f"{name}.json"), **self.conf_prediction)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_height.parameters(), lr=self.conf_training["lr"])
        return [optimizer]
