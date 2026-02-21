import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.handlers.utils import from_engine

class BraTSSegmentationModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1. Arquitectura de la Red: 3D U-Net estándar
        self.net = UNet(
            spatial_dims=3, # Fundamental: 3D
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            strides=cfg.strides,
            num_res_units=2,
            norm="batch",
        )

        # 2. Función de pérdida híbrida
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

        # 3. Métrica Clínica
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # 4. Post-procesamiento para las métricas
        self.post_pred = AsDiscrete(argmax=True, to_onehot=cfg.out_channels)
        self.post_label = AsDiscrete(to_onehot=cfg.out_channels)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        
        loss = self.loss_function(outputs, labels)
        
        # Logging de la pérdida
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        En validación, recibimos el volumen completo. Usamos sliding_window_inference
        para que la red extraiga parches, los prediga y ensamble el volumen final
        sin colapsar la VRAM de la GPU.
        """
        images, labels = batch["image"], batch["label"]
        
        # Inferencia por ventana deslizante
        roi_size = self.cfg.roi_size
        sw_batch_size = self.cfg.sw_batch_size
        outputs = sliding_window_inference(
            inputs=images, 
            roi_size=roi_size, 
            sw_batch_size=sw_batch_size, 
            predictor=self.forward
        )
        
        loss = self.loss_function(outputs, labels)
        
        # Post-procesamiento para calcular el Dice (One-hot encoding)
        outputs_list = [self.post_pred(i) for i in outputs]
        labels_list = [self.post_label(i) for i in labels]
        
        # Actualizamos el estado de la métrica
        self.dice_metric(y_pred=outputs_list, y=labels_list)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Calculamos el Dice medio de toda la época y reseteamos
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        self.log("val/dice_score", mean_dice, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # AdamW es un estándar muy sólido hoy en día
        optimizer = torch.optim.AdamW(
            self.net.parameters(), 
            lr=self.cfg.learning_rate, 
            weight_decay=self.cfg.weight_decay
        )
        
        # Un scheduler para ir reduciendo el learning rate ayuda a converger mejor
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            }
        }