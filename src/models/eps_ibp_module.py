# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import time
from typing import Any, List, Optional, Tuple

import pyrootutils
import torch
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src import utils  # noqa: E402
from src.models.base_bounded_module import BaseBoundedModulePL  # noqa: E402
from src.utils.scheduler import EpsScheduler  # noqa: E402

log = utils.get_pylogger(__name__)


class EpsIBP(BaseBoundedModulePL):
    """Model designed to certify robustness."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        input_shape: Tuple[int] = (1, 28, 28),
        num_classes: int = 10,
        data_mean: Tuple[float] = (0.0,),
        data_std: Tuple[float] = (1.0,),
        bound_norm: float = float("inf"),
        bound_method: str = "IBP",
        max_eps: float = 0.4,
        xtol: float = 2e-12,
        rtol: float = 8.882e-16,
        maxiter: int = 100,
        ftol: float = 0.0,
        standard_loss: str = "CrossEntropy",
        standard_label_smoothing: float = 0.0,
        robust_loss: str = "CrossEntropy",
        robust_label_smoothing: float = 0.0,
        eps_test: Tuple[float] = (0.3, 0.4),
        init_method: Optional[str] = "ibp",
        plot_train_val: bool = False,
        num_stn_epochs: int = 1,
        num_reg_epochs: int = 80,
        weight_decay_l1: float = 0.0,
        min_eps_for_regularizers: Optional[float] = 1e-6,
        reg_tight_weight: float = 0.5,
        reg_relu_weight: float = 0.5,
        kappa: float = 0.0,
        reg_clip_with_robust: bool = True,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            input_shape=input_shape,
            num_classes=num_classes,
            data_mean=data_mean,
            data_std=data_std,
            bound_norm=bound_norm,
            bound_method=bound_method,
            max_eps=max_eps,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
            ftol=ftol,
            standard_loss=standard_loss,
            standard_label_smoothing=standard_label_smoothing,
            robust_loss=robust_loss,
            robust_label_smoothing=robust_label_smoothing,
            eps_test=eps_test,
            init_method=init_method,
        )
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.eps_scheduler = EpsScheduler(
            schedule_start=num_stn_epochs + 1,
            schedule_length=num_reg_epochs,
            max_eps=max_eps,
        )

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_eps_best.reset()
        self.val_acceps_best.reset()

        dataset_len = len(self.trainer.train_dataloader.loaders.dataset)
        batch_size = self.trainer.train_dataloader.loaders.batch_size
        epoch_length = int((dataset_len + batch_size - 1) / batch_size)
        self.eps_scheduler.set_epoch_length(epoch_length)

    def model_step(self, batch: Any, xtol=None, rtol=None, maxiter=None, ftol=None):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss_stn = self.criterion(logits, y)

        step = self.trainer.global_step
        epoch = self.trainer.current_epoch

        eps_scalar_max = self.eps_scheduler.get_eps(step=step)

        # # Note: just for ablation study
        # eps_scalar_max = self.hparams.max_eps
        # ##############################

        # # Note: just for ablation study
        # idx_mis = preds.detach() != y
        # eps = eps_scalar_max * torch.ones_like(y).float().detach()
        # eps[idx_mis] = 0.0
        # fn_calls = torch.zeros_like(y).float().detach()
        ################################

        # # compute eps: R(x, eps) = 0
        with torch.no_grad():
            eps, fn_calls = self.find_eps(
                x,
                y,
                max_eps=eps_scalar_max,
                root_solver="brentq",
                xtol=xtol,
                rtol=rtol,
                maxiter=maxiter,
                ftol=ftol,
                logits=logits,
            )

        loss_rob = torch.tensor(0.0).to(x.device)
        loss_tight = torch.tensor(0.0).to(x.device)
        loss_relu = torch.tensor(0.0).to(x.device)

        reg = epoch < self.hparams.num_stn_epochs + self.hparams.num_reg_epochs

        # eps max cap
        eps = eps.clamp(max=eps_scalar_max)
        robust = eps_scalar_max >= 1e-50
        reg_coef = eps_scalar_max / self.hparams.max_eps

        if self.hparams.min_eps_for_regularizers is not None and reg:
            if self.hparams.reg_clip_with_robust:
                eps = eps.clamp(min=self.hparams.min_eps_for_regularizers)
            elif not robust:
                eps = eps.clamp(min=self.hparams.min_eps_for_regularizers)

        lb = self.compute_lb(x, y, eps=eps)
        loss_rob = self.robust_criterion(lb)

        loss = torch.tensor(0.0).to(x.device)
        if reg:
            loss_tight, loss_relu = self.compute_reg()
            loss += (1.0 - reg_coef) * (
                self.hparams.reg_tight_weight * loss_tight
                + self.hparams.reg_relu_weight * loss_relu
            )
        if robust:
            loss += (
                loss_rob * (1.0 - self.hparams.kappa) + loss_stn * self.hparams.kappa
            )
        else:
            loss += loss_stn

        if self.hparams.weight_decay_l1 > 0.0:
            loss_l1 = self.hparams.weight_decay_l1 * self.compute_l1_weight_decay()
            loss += loss_l1
        return loss, preds, y, eps, loss_stn, loss_rob, loss_tight, loss_relu, fn_calls

    def training_step(self, batch: Any, batch_idx: int):
        self.net.train()
        self.bounded_net.train()
        log_kwargs = {"on_step": False, "on_epoch": True, "prog_bar": True}
        start = time.time()
        (
            loss,
            preds,
            targets,
            eps,
            loss_stn,
            loss_rob,
            loss_tight,
            loss_relu,
            fn_calls,
        ) = self.model_step(
            batch,
            xtol=self.hparams.xtol,
            rtol=self.hparams.rtol,
            maxiter=self.hparams.maxiter,
            ftol=self.hparams.ftol,
        )

        end = time.time()
        self.log("step_time/train", end - start, **log_kwargs)
        self.log("train/loss_relu", loss_relu, **log_kwargs)
        self.log("train/loss_tight", loss_tight, **log_kwargs)
        # update and log metrics
        self.train_loss(loss)
        train_acc = self.train_acc(preds, targets)
        self.train_stn_loss(loss_stn)
        self.train_robust_loss(loss_rob)
        self.train_eps(eps.mean() / self.hparams.max_eps)
        self.train_acceps(torch.sqrt(eps.mean() / self.hparams.max_eps * train_acc))
        self.train_fn_calls(fn_calls.mean())

        self.log("train/loss", self.train_loss, **log_kwargs)
        self.log("train/acc", self.train_acc, **log_kwargs)
        self.log("train/loss_stn", self.train_stn_loss, **log_kwargs)
        self.log("train/loss_rob", self.train_robust_loss, **log_kwargs)
        self.log("train/eps", self.train_eps, **log_kwargs)
        self.log("train/acceps", self.train_acceps, **log_kwargs)
        self.log("train/fn_calls", self.train_fn_calls, **log_kwargs)

        for eps_t in self.hparams.eps_test:
            cert = self.compute_cert_acc(batch, eps_t)
            # get a metric
            self.__getattr__(f"train_cert_acc_{eps_t:.4f}")(cert)
            # log metric
            self.log(
                f"train/cert_acc_{eps_t:.4f}",
                self.__getattr__(f"train_cert_acc_{eps_t:.4f}"),
                **log_kwargs,
            )

        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "eps": eps}

    def update_state_dict_with_bn(self):
        bounded_state_dict = self.bounded_net.state_dict()
        state_dict = self.net.state_dict()
        keys = self.net.state_dict().keys()
        for name in bounded_state_dict:
            v = bounded_state_dict[name]
            for prefix in ["model.", "/w.", "/b.", "/running_mean."]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            if name not in keys:
                raise KeyError(name)
            # copy bounded_state_dict into state_dict because running_mean
            # and running_var are computed differently
            state_dict[name] = v
        self.net.load_state_dict(state_dict)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # if batch_norm is used, need to update state dict specifically:
        self.update_state_dict_with_bn()

    def validation_step(self, batch: Any, batch_idx: int):
        self.net.eval()
        self.bounded_net.eval()
        log_kwargs = {"on_step": False, "on_epoch": True, "prog_bar": True}
        start = time.time()
        (
            loss,
            preds,
            targets,
            eps,
            loss_stn,
            loss_rob,
            loss_tight,
            loss_relu,
            fn_calls,
        ) = self.model_step(batch, xtol=1e-8, rtol=1e-5, maxiter=100, ftol=1e-8)
        end = time.time()
        self.log("step_time/val", end - start, **log_kwargs)
        self.log("val/loss_relu", loss_relu, **log_kwargs)
        self.log("val/loss_tight", loss_tight, **log_kwargs)
        # update and log metrics
        self.val_loss(loss)
        val_acc = self.val_acc(preds, targets)
        self.val_stn_loss(loss_stn)
        self.val_robust_loss(loss_rob)
        self.val_eps(eps.mean() / self.hparams.max_eps)
        self.val_acceps(torch.sqrt(eps.mean() / self.hparams.max_eps * val_acc))
        self.val_fn_calls(fn_calls.mean())

        self.log("val/loss", self.val_loss, **log_kwargs)
        self.log("val/acc", self.val_acc, **log_kwargs)
        self.log("val/loss_stn", self.val_stn_loss, **log_kwargs)
        self.log("val/loss_rob", self.val_robust_loss, **log_kwargs)
        self.log("val/eps", self.val_eps, **log_kwargs)
        self.log("val/acceps", self.val_acceps, **log_kwargs)
        self.log("val/fn_calls", self.val_fn_calls, **log_kwargs)

        for eps_t in self.hparams.eps_test:
            cert = self.compute_cert_acc(batch, eps_t)
            # get a metric
            self.__getattr__(f"val_cert_acc_{eps_t:.4f}")(cert)
            # log metric
            self.log(
                f"val/cert_acc_{eps_t:.4f}",
                self.__getattr__(f"val_cert_acc_{eps_t:.4f}"),
                **log_kwargs,
            )

        return {"loss": loss, "eps": eps.detach()}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        eps = self.val_eps.compute()
        self.val_eps_best(eps)
        self.log("val/eps_best", self.val_eps_best.compute(), prog_bar=True)
        acceps = self.val_acceps.compute()
        self.val_acceps_best(acceps)
        self.log("val/acceps_best", self.val_acceps_best.compute(), prog_bar=True)
