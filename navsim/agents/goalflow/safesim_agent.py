"""
Original SafeSim Lightning agent used in the first training round.

Training and validation only optimize imitation-style flow-matching loss.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

from navsim.agents.goalflow.safesim_config import SafeSimConfig
from navsim.agents.goalflow.safesim_model import SafeSimModel


class SafeSimAgent(pl.LightningModule):
    """Lightning wrapper for the original SafeSim flow-matching model."""

    def __init__(self, config: SafeSimConfig):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self.model = SafeSimModel(config)
        self._transfer_unfrozen = config.freeze_loaded_fm_epochs == 0

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(features)

    def _compute_loss(self, predictions: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pred = predictions['trajectory']
        target = predictions['target']
        traj_loss = F.l1_loss(pred, target)
        loss_dict = {
            'trajectory_loss': traj_loss,
        }

        total_loss = self._config.trajectory_weight * traj_loss

        pred_future = predictions.get('predicted_future_trajectory')
        target_future = predictions.get('target_future_trajectory')
        ctrl_future = batch.get('ctrl_future')

        terminal_xy_loss = traj_loss.new_zeros(())
        if (
            self._config.terminal_xy_weight > 0.0
            and pred_future is not None
            and target_future is not None
        ):
            terminal_xy_loss = F.l1_loss(pred_future[:, -1, :2], target_future[:, -1, :2])
            total_loss = total_loss + self._config.terminal_xy_weight * terminal_xy_loss
        loss_dict['terminal_xy_loss'] = terminal_xy_loss

        terminal_heading_loss = traj_loss.new_zeros(())
        if (
            self._config.terminal_heading_weight > 0.0
            and pred_future is not None
            and target_future is not None
        ):
            heading_delta = pred_future[:, -1, 2] - target_future[:, -1, 2]
            heading_delta = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))
            terminal_heading_loss = heading_delta.abs().mean()
            total_loss = total_loss + self._config.terminal_heading_weight * terminal_heading_loss
        loss_dict['terminal_heading_loss'] = terminal_heading_loss

        ctrl_softmin_loss = traj_loss.new_zeros(())
        if (
            self._config.ctrl_softmin_weight > 0.0
            and pred_future is not None
            and ctrl_future is not None
        ):
            distances = torch.norm(pred_future[:, :, :2] - ctrl_future[:, :, :2], dim=-1)
            beta = max(self._config.ctrl_softmin_beta, 1e-6)
            ctrl_softmin_loss = (-torch.logsumexp(-beta * distances, dim=-1) / beta).mean()
            total_loss = total_loss + self._config.ctrl_softmin_weight * ctrl_softmin_loss
        loss_dict['ctrl_softmin_loss'] = ctrl_softmin_loss

        loss_dict['loss'] = total_loss
        return loss_dict

    def training_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        loss_dict = self._compute_loss(predictions, batch)

        self.log(
            'train_loss',
            loss_dict['loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch['drivable_map'].shape[0],
        )
        for key, value in loss_dict.items():
            self.log(
                f'train/{key}',
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch['drivable_map'].shape[0],
            )
        return loss_dict['loss']

    def on_train_epoch_start(self):
        if (
            not self._transfer_unfrozen
            and self._config.freeze_loaded_fm_epochs > 0
            and self.current_epoch >= self._config.freeze_loaded_fm_epochs
        ):
            self.model.unfreeze_loaded_transfer_modules()
            self._transfer_unfrozen = True

    def validation_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        loss_dict = self._compute_loss(predictions, batch)

        self.log(
            'val_loss',
            loss_dict['loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch['drivable_map'].shape[0],
        )
        for key, value in loss_dict.items():
            self.log(
                f'val/{key}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch['drivable_map'].shape[0],
            )
        return loss_dict['loss']

    def test_step(self, batch, batch_idx):
        orig_training = self._config.training
        self._config.training = False
        self.model._config.training = False
        try:
            return self.forward(batch)
        finally:
            self._config.training = orig_training
            self.model._config.training = orig_training

    def configure_optimizers(self):
        # All params are owned by the optimizer; freeze is enforced via requires_grad.
        # When a param is frozen, autograd writes no .grad and Adam skips its update,
        # so toggling requires_grad later via on_train_epoch_start works without rebuilding.
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._config.lr,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self._config.step_size,
            gamma=self._config.gamma,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
