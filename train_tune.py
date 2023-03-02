from typing import Optional, Union, Dict

from train import BrainGNNTrain
import torch
from ray.tune import Trainable
from ray.tune.integration.wandb import (
    WandbTrainableMixin,
    wandb_mixin,
)
from dataclasses import dataclass
import numpy as np
import os
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrialInfo:
    """The trial information to propagate to TrainSession."""

    name: str
    id: str
    resources: Dict[str, float]
    logdir: str


class TuneTrain(BrainGNNTrain, WandbTrainableMixin, Trainable):
    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        torch.load(checkpoint + "/model.pt")

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model.state_dict(), checkpoint_dir + "/model.pt")
        return None

    def __init__(self, config, model, optimizers, dataloaders, log_folder, session):
        train_config = config['train']
        super(TuneTrain, self).__init__(train_config, model, optimizers, dataloaders, log_folder)
        Trainable.__init__(self, config=config)
        # Get current pwd
        cwd = os.getcwd()
        # get folder name
        folder_name = os.path.basename(cwd)
        # First 128 characters of the folder name
        train_config["wandb"]["group"] = folder_name[:128]
        config["wandb"] = train_config["wandb"]

        wandb_init_kwargs = dict(
            id=session.get_trial_id(),
            name=session.get_trial_name(),
            resume=True,
            reinit=True,
            allow_val_change=True,
            group=config["wandb"]["group"],
            project=config["wandb"]["project"],
            config=config,
        )
        self.wandb = wandb.init(**wandb_init_kwargs)
        print("Wandb init complete")
        print("Current working directory: ", )
        self.save_learnable_graph = False
        self.diff_loss = train_config.get('diff_loss', False)
        self.cluster_loss = train_config.get('cluster_loss', True)
        self.assignment_loss = train_config.get('assignment_loss', True)
        self.session = session
        self.epoch = 0
        self.wandb.tags += ("ray_tune",)
        self.wandb.tags += (train_config["wandb"]["dataset"],)

    def step(self):
        self.reset_meters()
        self.train_per_epoch(self.optimizers[0])
        val_result = self.test_per_epoch(self.val_dataloader,
                                         self.val_loss, self.val_accuracy)

        test_result = self.test_per_epoch(self.test_dataloader,
                                          self.test_loss, self.test_accuracy)

        self.logger.info(" | ".join([
            f'Epoch[{self.epoch}/{self.epochs}]',
            f'Train Loss:{self.train_loss.avg: .3f}',
            f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
            f'Edges:{self.edges_num.avg: .3f}',
            f'Test Loss:{self.test_loss.avg: .3f}',
            f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
            f'Val AUC:{val_result[0]:.4f}',
            f'Test AUC:{test_result[0]:.4f}'
        ]))

        metrics = {
            "Epoch": self.epoch,
            "Train Loss": self.train_loss.avg,
            "Train Accuracy": self.train_accuracy.avg,
            "Test Loss": self.test_loss.avg,
            "Test Accuracy": self.test_accuracy.avg,
            "Val AUC": val_result[0],
            "Test AUC": test_result[0]
        }

        return metrics

    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            metrics = self.step()
            self.wandb.log(metrics)
            self.session.report(metrics)

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()

    def stop(self):
        self.wandb.run.finish()
        if hasattr(super(), "stop"):
            super().stop()
