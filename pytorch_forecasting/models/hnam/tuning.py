"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import copy
import logging
import os
from typing import Any, Dict, Tuple, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import optuna.logging
import statsmodels.api as sm
import torch
from torch.utils.data import DataLoader

from pytorch_forecasting import NAM_TS
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import math

class BestEpochEarlyStopping(EarlyStopping):    
   
    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        self.best_epoch = state_dict["best_epoch"]

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current,trainer)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria(self, current: Tensor,trainer) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
            self.best_epoch = trainer.current_epoch
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

optuna_logger = logging.getLogger("optuna")


# need to inherit from callback for this to work
class PyTorchLightningPruningCallbackAdjusted(pl.Callback, PyTorchLightningPruningCallback):
    pass


def optimize_hyperparameters(
    train_dataloaders: DataLoader,
    val_dataloaders: DataLoader,
    model_path: str,
    max_epochs: int = 20,
    n_trials: int = 100,
    timeout: float = 3600 * 8.0,  # 8 hours

    ### GPT, ATTEND TO THE NEW PARAMETERS AND REPLACE THE ARGUMENTS BELOW
    gradient_clip_val_range: Tuple[float, float] = (0.01, 0.1),
    learning_rate_range: Tuple[float, float] = (1e-5, 1.0), 
    use_learning_rate_finder: bool = True,
    patience: int = 6,
    trainer_kwargs: Dict[str, Any] = {},
    log_dir: str = "lightning_logs",
    study: optuna.Study = None,
    verbose: Union[int, bool] = None,
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
    **kwargs,
) -> optuna.Study:

    assert isinstance(train_dataloaders.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloaders.dataset, TimeSeriesDataSet
    ), "dataloaders must be built from timeseriesdataset"

    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    loss = kwargs.get(
        "loss", QuantileLoss()
    )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next

    # create objective function
    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(model_path, "trial_{}".format(trial.number)),
            filename='best-{val_loss:.4f}-{epoch:02d}-{step:02d}',
            save_top_k=1,
            monitor="val_loss"
        )

        learning_rate_callback = LearningRateMonitor()
        early_stop_callback = BestEpochEarlyStopping(
                                monitor='val_loss',  # val_SMAPE
                                min_delta=1e-4,
                                patience=patience,
                                verbose=True,  
                                mode='min'
                            )
        logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
        gradient_clip_val = trial.suggest_float("gradient_clip_val", *gradient_clip_val_range,log=True)
        default_trainer_kwargs = dict(
            accelerator='cpu',
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                early_stop_callback,
                learning_rate_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallbackAdjusted(trial, monitor="val_loss"),
            ],
            logger=logger,
            enable_progress_bar=optuna_verbose < optuna.logging.INFO,
            enable_model_summary=[False, True][optuna_verbose < optuna.logging.INFO],
        )
        default_trainer_kwargs.update(trainer_kwargs)
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )

        # create model
        loss = QuantileLoss([0.5])
        kwargs["loss"] = copy.deepcopy(loss)


        config = {
            'cov_emb': trial.suggest_categorical("cov_emb", [8, 16, 32,64]),
            'trend_emb': trial.suggest_categorical("trend_emb", [4,8,16]),
            'max_cat_emb': trial.suggest_categorical("max_cat_emb", [4, 8]),
            'time_emb': trial.suggest_categorical("time_emb", [False]),
            'cov_heads': trial.suggest_categorical("cov_heads", [1,2,4]),
            'dropout': trial.suggest_float("dropout", 0.1, 0.3),
            'factor': trial.suggest_categorical("factor", [1, 2]),
            'pre_blocks': trial.suggest_int("pre_blocks", 0, 3),
            'bias': trial.suggest_categorical("bias", [False]),
            'trend_query': trial.suggest_categorical("trend_query", [['time_idx', 'weekofyear']]),
            'exclude_cat': trial.suggest_categorical("exclude_cat", [['weekofyear']]),
            'exclude_cont': trial.suggest_categorical("exclude_cont", [['time_idx', 'relative_time_idx']])
        }

        model = NAM_TS.from_dataset(
            train_dataloaders.dataset,
            **config,
            log_interval=-1,
            **kwargs,
        )
        # find good learning rate
        if use_learning_rate_finder:
            lr_trainer = pl.Trainer(
                gradient_clip_val=gradient_clip_val,
                accelerator='cpu',
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            tuner = Tuner(lr_trainer)
            res = tuner.lr_find(
                model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                early_stop_threshold=10000,
                min_lr=learning_rate_range[0],
                num_training=100,
                max_lr=learning_rate_range[1],
            )

            loss_finite = np.isfinite(res.results["loss"])
            if loss_finite.sum() > 3:  # at least 3 valid values required for learning rate finder
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[min(loss_finite.sum() - 3, 10) : -1].T
                optimal_idx = np.gradient(loss_smoothed).argmin()
                optimal_lr = lr_smoothed[optimal_idx]
            else:
                optimal_idx = np.asarray(res.results["loss"]).argmin()
                optimal_lr = res.results["lr"][optimal_idx]
            optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
            # add learning rate artificially
            model.hparams.learning_rate = trial.suggest_float("learning_rate", optimal_lr, optimal_lr)
        else:
            model.hparams.learning_rate = trial.suggest_float("learning_rate", *learning_rate_range,log=True)

        # fit
        trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)


        # Report the best validation loss
        print(trainer.callback_metrics["val_loss"].item())
        trial.set_user_attr("best_epoch", early_stop_callback.best_epoch)
        return trainer.checkpoint_callback.best_model_score.item()

    # setup optuna and run
    if study is None:
        study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
