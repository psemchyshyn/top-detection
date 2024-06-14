import torch
import os
import numpy as np
import skimage
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import SamModel, SamProcessor
from utility import mask2labelme
from scorer import get_score
import monai


class LitSegment(pl.LightningModule):
    def __init__(self, config):
        super(LitSegment, self).__init__()
        self.save_hyperparameters()

        self.config = self.hparams['config']
        self.conf_training = self.config['training']
        self.conf_prediction = self.config['prediction']
        self.conf_data = self.config['data']

        torch.hub.set_dir(self.conf_training['cache_dir'])

        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge")

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
                

        self.loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.test_results_folder = self.conf_training['save_results_dir']

    def check_if_augs_present(self, mode):
        return self.conf_data[f'{mode}_augs'] in ['easy', 'medium', 'hard']
    
    def get_bounding_box(self, ground_truth_map):
        # get bounding box from mask
        ground_truth_map = ground_truth_map.detach().squeeze(0).cpu().numpy()

        # bboxes = []

        # connected_regions = skimage.measure.label(ground_truth_map)
        # max_label = connected_regions.max()
        # for i in range(1, max_label + 1):
        #     y_indices, x_indices = np.where(connected_regions == i)

        #     x_min, x_max = np.min(x_indices), np.max(x_indices)
        #     y_min, y_max = np.min(y_indices), np.max(y_indices)
        #     # add perturbation to bounding box coordinates
        #     H, W = ground_truth_map.shape
        #     x_min = max(0, x_min - np.random.randint(0, 20))
        #     x_max = min(W, x_max + np.random.randint(0, 20))
        #     y_min = max(0, y_min - np.random.randint(0, 20))
        #     y_max = min(H, y_max + np.random.randint(0, 20))
        #     bbox = [x_min, y_min, x_max, y_max]
        #     bboxes.append(bbox)


        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return [bbox]
    
    def forward_roof_train(self, x, mask_roof):
        # prompt = [0, 0, self.conf_data['image_h'], self.conf_data['image_w']]
        # inputs = self.processor(x, input_boxes=[[prompt]]*self.conf_data['batch_size'], return_tensors="pt")
        boxes = [self.get_bounding_box(m) for m in mask_roof]
        inputs = self.processor(x, input_boxes=boxes, return_tensors="pt")
        outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device),
                        input_boxes=inputs["input_boxes"].to(self.device),
                        multimask_output=False)

        masks = outputs.pred_masks.squeeze(1)
        masks = self.resize_if_needed(masks)
        return masks
    

    def get_input_points(self):
        array_size = self.conf_data['image_w']

        # Define the size of your grid
        grid_size = 10

        # Generate the grid points
        x = np.linspace(0, array_size-1, grid_size)
        y = np.linspace(0, array_size-1, grid_size)

        # Generate a grid of coordinates
        xv, yv = np.meshgrid(x, y)

        # Convert the numpy arrays to lists
        xv_list = xv.tolist()
        yv_list = yv.tolist()

        # Combine the x and y coordinates into a list of list of lists
        input_points = np.array([[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)])

        input_points = torch.tensor(input_points).view(1, grid_size*grid_size, 2)  

        return input_points 
     
    def forward_roof(self, x):
        points = torch.stack([self.get_input_points()]*x.shape[0])
        inputs = self.processor(x, return_tensors="pt")
        outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device),
                        input_points=points.to(self.device),
                        multimask_output=False)

        masks = outputs.pred_masks.squeeze(1)
        masks = self.resize_if_needed(masks)
        return masks

    def forward_height(self, y_roof):
        y_height = (y_roof > 0.5).float()
        y_height[y_height > 0.5] = self.conf_training['set_height_to']

        return y_height
    
    def resize_if_needed(self, tensor):
        if tensor.shape[-1] != self.conf_data['original_image_size']:
            tensor = F.interpolate(tensor, size=(self.conf_data['original_image_size'], )*2, mode='bilinear', align_corners=False)
        return tensor
    
    def shared_step(self, batch, mode):
        x = batch["image"]
        mask_roof = batch["mask_roof"]
        mask_height = batch["mask_height"]


 
        y_hat_roof = self.forward_roof(x)
        y_hat_height = self.forward_height(y_hat_roof)
        loss = self.loss(y_hat_roof, mask_roof.float())


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
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss
    
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
        y_hat_roof = self.forward_roof(x)
        y_hat_height = self.forward_height(y_hat_roof)
        y_hat_roof = (y_hat_roof > 0.5).float()
        self.predict_batch(names, y_hat_roof, y_hat_height, self.test_results_folder, self.conf_prediction)

        if 'mask_roof' in batch and 'mask_height' in batch:
            self.label_batch(names, batch['mask_roof'], batch['mask_height'], self.test_mode)

    def form_predicion(self, name, y_roof, y_height, output_folder, conf_prediction):
        pred_mask_roof = y_roof.squeeze(0).cpu().detach().numpy()
        pred_mask_height = y_height.squeeze(0).cpu().detach().numpy() 

        mask2labelme(pred_mask_roof, pred_mask_height, os.path.join(output_folder, f"{name}.json"), **conf_prediction)
        return pred_mask_roof
    
    def predict(self, element, output_folder, conf_prediction):
        x = element['image'].cuda()
        name = element['name']
        y_roof = self.forward_roof(x).squeeze(0)
        y_height = self.forward_height(y_roof).squeeze(0) 

        y_roof = (y_roof > 0.5).float()
 
        return self.form_predicion(name, y_roof, y_height, output_folder, conf_prediction)
        
    def predict_batch(self, names, y_roofs, y_heights, output_dir, conf_prediction):
        for name, y_roof, y_height in zip(names, y_roofs, y_heights):
            self.form_predicion(name, y_roof, y_height, output_dir, conf_prediction)

    def label_batch(self, names, masks_roof, masks_height, mode='train'):
        for name, m_roof, m_height in zip(names, masks_roof, masks_height):
            m_roof = m_roof.squeeze(0).cpu().detach().numpy()
            m_height = m_height.squeeze(0).cpu().detach().numpy() 
            mask2labelme(m_roof, m_height, os.path.join(self.conf_training[f'{mode}_labels_dir_gt'], f"{name}.json"), **self.conf_prediction)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=self.conf_training["lr"])
        return [optimizer]
