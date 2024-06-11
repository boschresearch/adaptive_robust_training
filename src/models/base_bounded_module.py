# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import math
import os
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundAdd, BoundConv, BoundLinear, BoundRelu
from auto_LiRPA.perturbations import PerturbationLpNorm
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src import utils  # noqa: E402
from src.utils import loss_fn, root_finder  # noqa: E402

log = utils.get_pylogger(__name__)


class BaseBoundedModulePL(LightningModule):
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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.lr = optimizer.lr
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self._init_bounded_module(input_shape)
        if init_method is not None:
            self._manual_init(init_method=init_method)

        self.criterion = self._init_loss(
            standard_loss, standard_label_smoothing, robust=False
        )
        self.robust_criterion = self._init_loss(
            robust_loss,
            robust_label_smoothing,
            robust=True,
        )

        self.train_stn_loss = MeanMetric()
        self.val_stn_loss = MeanMetric()
        self.test_stn_loss = MeanMetric()
        self.train_robust_loss = MeanMetric()
        self.val_robust_loss = MeanMetric()
        self.test_robust_loss = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_eps = MeanMetric()
        self.val_eps = MeanMetric()
        self.test_eps = MeanMetric()
        self.train_acceps = MeanMetric()
        self.val_acceps = MeanMetric()
        self.test_acceps = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.val_eps_best = MaxMetric()
        self.val_acceps_best = MaxMetric()

        self.train_fn_calls = MeanMetric()
        self.val_fn_calls = MeanMetric()
        self.test_fn_calls = MeanMetric()

        self.train_cert_acc = {}
        self.val_cert_acc = {}
        self.test_cert_acc = {}
        for eps_t in eps_test:
            self.__setattr__(f"train_cert_acc_{eps_t:.4f}", MeanMetric())
            self.__setattr__(f"val_cert_acc_{eps_t:.4f}", MeanMetric())
            self.__setattr__(f"test_cert_acc_{eps_t:.4f}", MeanMetric())

    def _init_bounded_module(self, input_shape=None):
        input_shape = self.hparams.input_shape if input_shape is None else input_shape
        dummy_input = torch.rand(
            2, *input_shape, device=self.device
        )  # floats in [0, 1)
        self.bounded_net = BoundedModule(
            self.net,
            dummy_input,
            bound_opts={"relu": "adaptive", "conv_mode": "patches"},
        )
        return self.bounded_net

    def _init_loss(
        self,
        name,
        label_smoothing=0.0,
        robust=False,
    ):
        if not robust:
            if name == "CrossEntropy":
                loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                log.error(
                    f"{name} is not defined for standard_loss. Use `CrossEntropy` instead."
                )
                return -1
        else:
            if name == "CrossEntropy":
                loss = loss_fn.RobustCrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                log.error(
                    f"{name} is not defined for robust_loss. Use `CrossEntropy` or `TRADESLoss` instead."
                )
                return -1
        return loss

    def forward(self, x: torch.Tensor):
        return self.bounded_net(x)

    def compute_lb(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eps: torch.Tensor,
        bound_method: Optional[str] = None,
    ) -> torch.Tensor:
        """Computes the lower bound of `f(x)_y - f(x)_t` for all `num_classes - 1` target classes `t`.
        If the lower bound > 0, then `f(x)_y > f(x)_t`, and there is a robustness certification.

        Args:
            x (torch.Tensor): batch of input vectors of shape (b, c, w, h)
            y (torch.Tensor): 1D-tensor of ground truth labels, shape = (b)
            eps (torch.Tensor): either 1D-tensor of perturbation radii, shape = (b), or a scalar tensor, shape = (1)
            bound_method (Optional[str], optional): Bounding method, see `method` in `BoundedModule.compute_bounds`. Defaults to None.

        Returns:
            torch.Tensor: lower bounds of `f(x)_y - f(x)_t`, shape = (b, num_classes - 1)
        """
        if bound_method is None:
            bound_method = self.hparams.bound_method
        # assuming that x_max is 1., and x_min is 0., find epsilon ball around x
        # assume that x is normalized by dataset characteristics, as `x = (x_un - mean) / std`
        data_mean = torch.tensor([*self.hparams.data_mean], device=self.device)
        data_std = torch.tensor([*self.hparams.data_std], device=self.device)
        x_un = x * data_std.view(1, -1, 1, 1) + data_mean.view(1, -1, 1, 1)
        if x_un.max().item() > 1.0 + 5.0e-8 or x_un.min().item() < -5.0e-8:
            # throw am error log
            log.error(
                (
                    f"The input data is not normalized into [0, 1] as expected!"
                    f"Max value = {x_un.max().item()}, min value = {x_un.min().item()}."
                )
            )

        eps_scaled = eps.view(-1, 1, 1, 1) / data_std.view(1, -1, 1, 1)
        if self.hparams.bound_norm == float("inf"):
            x_max = torch.reshape((1.0 - data_mean) / data_std, (1, -1, 1, 1))
            x_min = torch.reshape((0.0 - data_mean) / data_std, (1, -1, 1, 1))
            x_ub = torch.min(x + eps_scaled, x_max)
            x_lb = torch.max(x - eps_scaled, x_min)
        else:
            x_ub = x_lb = None
        ptb = PerturbationLpNorm(
            norm=self.hparams.bound_norm, eps=eps_scaled, x_L=x_lb, x_U=x_ub
        )
        bounded_x = BoundedTensor(x, ptb)

        # generate specifications: y - y_target
        spec_y = torch.eye(self.hparams.num_classes).type_as(x)[y].unsqueeze(
            1
        ) - torch.eye(self.hparams.num_classes).type_as(x).unsqueeze(0)
        # remove specifications to self
        idx_y = ~(
            y.data.unsqueeze(1)
            == torch.arange(self.hparams.num_classes).type_as(y.data).unsqueeze(0)
        )
        spec_y = spec_y[idx_y].view(
            x.size(0), self.hparams.num_classes - 1, self.hparams.num_classes
        )
        # compute lower bound of `p_y - p_target`` for all (num_classes - 1) targets.
        # if lb > 0, then `p_y > p_target` for all (num_classes - 1) targets.
        # lb.shape = (batch_size, num_classes - 1)
        lb, _ = self.bounded_net.compute_bounds(
            x=(bounded_x,), C=spec_y, method=bound_method
        )
        return lb

    def find_eps(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        max_eps: Optional[float] = None,
        bound_method: Optional[str] = None,
        root_solver: str = "brentq",
        xtol: Optional[float] = None,
        rtol: Optional[float] = None,
        maxiter: Optional[int] = None,
        ftol: Optional[float] = None,
        logits: Optional[torch.Tensor] = None,
    ):
        bound_method = (
            self.hparams.bound_method if bound_method is None else bound_method
        )
        xtol = self.hparams.xtol if xtol is None else xtol
        rtol = self.hparams.rtol if rtol is None else rtol
        maxiter = self.hparams.maxiter if maxiter is None else maxiter
        ftol = self.hparams.ftol if ftol is None else ftol
        max_eps = self.hparams.max_eps if max_eps is None else max_eps

        res_eps = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        fn_calls = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

        # filter out samples that are misclassified. For them the eps = 0.
        if logits is None:
            with torch.no_grad():
                logits = self.forward(x)
        preds = logits.detach().clone()
        idx_mis = torch.argmax(preds, dim=1) != y

        if torch.all(idx_mis):
            return res_eps, fn_calls
        # create fa
        # generate specifications: y - y_target
        n_class = self.hparams.num_classes
        with torch.no_grad():
            spec_y = torch.eye(n_class).type_as(x)[y].unsqueeze(1) - torch.eye(
                n_class
            ).type_as(x).unsqueeze(0)
            # remove specifications to self
            idx_y = ~(
                y.data.unsqueeze(1)
                == torch.arange(n_class).type_as(y.data).unsqueeze(0)
            )
            spec_y = spec_y[idx_y].view(x.size(0), n_class - 1, n_class)
            lb_zero = torch.bmm(spec_y, preds.unsqueeze(-1)).squeeze(-1)
            # we return `f(x)_t - f(x)_y`, so that the function is increasing wrt eps
            fa = -torch.min(lb_zero, dim=-1).values

            # filter out samples that are correctly classified with eps_max.
            eps_try = max_eps * torch.ones(x.size(0)).type_as(x)
            lb_try = self.compute_lb(x, y, eps_try, bound_method)
            fb = -torch.min(lb_try, dim=-1).values
            idx_mis = fa >= 0
            idx_robust = fb <= 0
            fn_calls[~idx_mis] = fn_calls[~idx_mis] + 1.0
        res_eps[idx_mis] = 0.0
        res_eps[idx_robust] = max_eps
        idx_find = torch.logical_and(~idx_mis, ~idx_robust)

        xfind = x.detach().clone()
        yfind = y.detach().clone()

        fbracket = (fa.detach().clone(), fb.detach().clone())
        xbracket = (
            torch.zeros(x.size(0)).type_as(x),
            max_eps * torch.ones(x.size(0)).type_as(x),
        )
        args = (
            xfind,
            yfind,
            bound_method,
            root_solver,
            xtol,
            rtol,
            maxiter,
            ftol,
            max_eps,
            xbracket,
            fbracket,
        )
        roots, fn_calls_solver = self._find_eps_pytorch(*args)
        res_eps[idx_find] = roots[idx_find].detach().clone()
        fn_calls[idx_find] = fn_calls[idx_find] + fn_calls_solver[idx_find]
        return res_eps, fn_calls

    def compute_cert_acc(
        self,
        batch: Any,
        eps: float,
        bound_method: Optional[str] = None,
    ):
        x, y = batch
        bound_method = (
            self.hparams.bound_method if bound_method is None else bound_method
        )
        eps_tensor = eps * torch.ones(x.size(0)).type_as(x)
        lb = self.compute_lb(x, y, eps=eps_tensor, bound_method=bound_method)
        cert = torch.all(lb > 0, dim=-1)
        return cert

    def compute_relu_loss(self, reg_tol=0.5):
        # call it after computing bounds, i.e. assuming upper and lower bounds are saved
        loss_relu = torch.zeros(()).to(self.device)
        cnt = 0
        for m in self.bounded_net._modules.values():
            if isinstance(m, BoundRelu):
                cnt += 1
                # L_{relu}
                lower, upper = m.inputs[0].lower, m.inputs[0].upper
                center = (upper + lower) / 2.0
                mean_ = center.mean()
                mask_act, mask_inact = lower > 0, upper < 0
                mean_act = (center * mask_act).mean()
                mean_inact = (center * mask_inact).mean()
                delta = (center - mean_) ** 2
                var_act = (delta * mask_act).sum()
                var_inact = (delta * mask_inact).sum()
                mean_ratio = mean_act / -mean_inact.clamp(min=1e-12)
                var_ratio = var_act / var_inact.clamp(min=1e-12)
                mean_ratio = torch.min(mean_ratio, 1.0 / mean_ratio.clamp(min=1e-12))
                var_ratio = torch.min(var_ratio, 1.0 / var_ratio.clamp(min=1e-12))
                loss_relu_ = (
                    F.relu(reg_tol - mean_ratio) + F.relu(reg_tol - var_ratio)
                ) / reg_tol
                if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                    loss_relu += loss_relu_
        loss_relu /= cnt
        return loss_relu

    def compute_tightness_loss(self, reg_tol=0.5):
        # call it after computing bounds, i.e. assuming upper and lower bounds are saved
        loss_tightness = torch.zeros(()).to(self.device)
        modules = self.bounded_net._modules
        node_inp = modules["/input.1"]
        tightness_0 = ((node_inp.upper - node_inp.lower) / 2.0).mean()
        cnt = 0
        for m in self.bounded_net._modules.values():
            if isinstance(m, BoundRelu):
                cnt += 1
                # L_{tightness}
                lower, upper = m.inputs[0].lower, m.inputs[0].upper
                diff = (upper - lower) / 2.0
                tightness = diff.mean()
                loss_tightness += (
                    F.relu(reg_tol - tightness_0 / tightness.clamp(min=1e-12)) / reg_tol
                )
        loss_tightness /= cnt
        return loss_tightness

    def compute_reg(self, tol=0.5):
        loss = torch.zeros(()).to(self.device)
        l0 = torch.zeros_like(loss)
        loss_tightness, loss_std, loss_relu, loss_ratio = (l0.clone() for i in range(4))

        modules = self.bounded_net._modules
        node_inp = modules["/input.1"]
        tightness_0 = ((node_inp.upper - node_inp.lower) / 2).mean()
        # ratio_init = tightness_0 / ((node_inp.upper + node_inp.lower) / 2).std()
        # cnt_layers = 0
        cnt = 0
        for m in self.bounded_net._modules.values():
            if isinstance(m, BoundRelu):
                lower, upper = m.inputs[0].lower, m.inputs[0].upper
                center = (upper + lower) / 2
                diff = (upper - lower) / 2
                tightness = diff.mean()
                mean_ = center.mean()
                std_ = center.std()

                loss_tightness += (
                    F.relu(tol - tightness_0 / tightness.clamp(min=1e-12)) / tol
                )
                loss_std += F.relu(tol - std_) / tol
                cnt += 1

                # L_{relu}
                mask_act, mask_inact = lower > 0, upper < 0
                mean_act = (center * mask_act).mean()
                mean_inact = (center * mask_inact).mean()
                delta = (center - mean_) ** 2
                var_act = (delta * mask_act).sum()  # / center.numel()
                var_inact = (delta * mask_inact).sum()  # / center.numel()

                mean_ratio = mean_act / -mean_inact
                var_ratio = var_act / var_inact
                mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
                var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
                loss_relu_ = (F.relu(tol - mean_ratio) + F.relu(tol - var_ratio)) / tol
                if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                    loss_relu += loss_relu_

        loss_tightness /= cnt
        loss_std /= cnt
        loss_relu /= cnt

        return loss_tightness, loss_relu

    def compute_l1_weight_decay(self):
        loss_l1 = torch.zeros(()).to(self.device)
        for module in self.net._modules.values():
            if isinstance(module, nn.Linear):
                loss_l1 += torch.abs(module.weight).sum()
            elif isinstance(module, nn.Conv2d):
                loss_l1 += torch.abs(module.weight).sum()
        return loss_l1

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        # Add explicitly net and bounded_net
        self.net.train(mode)
        self.bounded_net.train(mode)
        return self

    def model_step(self, batch: Any, xtol=None, rtol=None, maxiter=None, ftol=None):
        pass

    def on_train_start(self):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        self.net.eval()
        self.bounded_net.eval()
        log_kwargs = {"on_step": False, "on_epoch": True, "prog_bar": True}
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        with torch.no_grad():
            eps, fn_calls = self.find_eps(
                x,
                y,
                root_solver="brentq",
                logits=logits,
                xtol=1e-8,
                rtol=1e-5,
                maxiter=100,
                ftol=1e-8,
            )

        # update and log metrics
        test_acc = self.test_acc(preds, y)
        self.test_eps(eps.mean() / self.hparams.max_eps)
        self.test_acceps(torch.sqrt(eps.mean() / self.hparams.max_eps * test_acc))
        self.test_fn_calls(fn_calls.mean())

        self.log("test/acc", self.test_acc, **log_kwargs)
        self.log("test/eps", self.test_eps, **log_kwargs)
        self.log("test/acceps", self.test_acceps, **log_kwargs)
        self.log("test/fn_calls", self.test_fn_calls, **log_kwargs)

        for eps_t in self.hparams.eps_test:
            cert = self.compute_cert_acc(batch, eps_t)
            # get a metric
            self.__getattr__(f"test_cert_acc_{eps_t:.4f}")(cert)
            # log metric
            self.log(
                f"test/cert_acc_{eps_t:.4f}",
                self.__getattr__(f"test_cert_acc_{eps_t:.4f}"),
                **log_kwargs,
            )

        return {"eps": eps.detach()}

    def test_epoch_end(self, outputs: List[Any]):
        eps_all = np.array([])
        for output in outputs:
            eps_all = np.concatenate((eps_all, output["eps"].cpu().numpy()))
        dir = os.path.join(self.trainer._default_root_dir, "figures", "test")
        epoch = self.trainer.current_epoch
        file_name = f"eps_test_{epoch:05}.npy"
        os.makedirs(dir, exist_ok=True)
        np.save(os.path.join(dir, file_name), eps_all)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        opt_cls = getattr(torch.optim, self.hparams.optimizer.name)
        optimizer = opt_cls(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )
        if self.hparams.lr_scheduler is not None:
            lr_cls = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler.name)
            scheduler_kwargs = self.hparams.lr_scheduler.copy()
            del scheduler_kwargs["name"]
            del scheduler_kwargs["interval"]
            lr_scheduler = lr_cls(optimizer, **scheduler_kwargs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": self.hparams.lr_scheduler.interval,  # "step",  # epoch
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state = self.net.state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.net.eval()
        self.bounded_net.eval()
        ret = self.net.load_state_dict(state_dict, strict)
        log.info("Note: loaded state_dict for model.net.")
        self._init_bounded_module()
        self.bounded_net.eval()
        log.info("Note: Re-initialized bounded_net with loaded weights for model.net")
        return ret

    def load_state_dict_from_checkpoint(self, path, strict=False):
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"]
        log.info(f"Checkpoint loaded from {path}")
        ret = self.load_state_dict(state_dict, strict=strict)
        return ret

    def _find_eps_pytorch(
        self,
        x,
        y,
        bound_method,
        root_solver,
        xtol,
        rtol,
        maxiter,
        ftol,
        max_eps=0.5,
        xbracket=None,
        fbracket=None,
    ):
        def eval_func(eps_, x_, y_):
            with torch.no_grad():
                lb = self.compute_lb(x_, y_, eps_, bound_method)
                out = -torch.min(lb, dim=-1).values
                # we return `f(x)_t - f(x)_y`, so that the function is increasing wrt eps
                return out

        if xbracket is None:
            a = torch.zeros(x.size(0)).type_as(x)
            b = max_eps * torch.ones(x.size(0)).type_as(x)
            xbracket = (a, b)
        solver = root_finder.PytorchOptimizeRootScalarBatch(
            xtol=xtol, rtol=rtol, maxiter=maxiter, ftol=ftol
        )
        roots, solver_outs = solver.solve(
            f=eval_func,
            args=(x, y),
            method=root_solver,
            bracket=xbracket,
            fbracket=fbracket,
        )
        roots = roots.type_as(x).to(x.device)
        fn_calls = solver_outs.function_calls.type_as(x).to(x.device)
        return roots, fn_calls

    def _get_params(self):
        weights = []
        biases = []
        for p in self.net.named_parameters():
            if "weight" in p[0]:
                weights.append(p)
            elif "bias" in p[0]:
                biases.append(p)
            else:
                print("Skipping parameter {}".format(p[0]))
        return weights, biases

    def _ibp_init(self):
        weights, biases = self._get_params()
        for i in range(len(weights) - 1):
            if weights[i][1].ndim == 1:
                continue
            weight = weights[i][1]
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            std = math.sqrt(2 * math.pi / (fan_in**2))
            std_before = weight.std().item()
            torch.nn.init.normal_(weight, mean=0, std=std)
            log.debug(
                f"Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}"
            )
        for node in self.bounded_net._modules.values():
            if isinstance(node, BoundConv) or isinstance(node, BoundLinear):
                if len(node.inputs[0].inputs) > 0 and isinstance(
                    node.inputs[0].inputs[0], BoundAdd
                ):
                    log.debug(
                        f"Adjust weights for node {node.name} due to residual connection"
                    )
                    node.inputs[1].param.data /= 2

    def _kaiming_init(self):
        for p in self.net.named_parameters():
            if p[0].find(".weight") != -1:
                if p[0].find("bn") != -1 or p[1].ndim == 1:
                    continue
                torch.nn.init.kaiming_normal_(p[1].data)

    def _orthogonal_init(self):
        params = []
        bns = []
        for p in self.net.named_parameters():
            if p[0].find(".weight") != -1:
                if p[0].find("bn") != -1 or p[1].ndim == 1:
                    bns.append(p)
                else:
                    params.append(p)
        for p in params[:-1]:
            std_before = p[1].std().item()
            log.debug("before mean abs", p[1].abs().mean())
            torch.nn.init.orthogonal_(p[1])
            log.debug(
                "Reinitialize {} with orthogonal matrix, std before {:.5f}, std now {:.5f}".format(
                    p[0], std_before, p[1].std()
                )
            )
            log.debug("after mean abs", p[1].abs().mean())

    def _manual_init(self, init_method: Optional[str] = None):
        init_method = self.hparams.init_method if init_method is None else init_method
        if init_method.lower() == "ibp":
            log.info("Initialization: IBP normal")
            self._ibp_init()
        elif init_method.lower() == "orthogonal":
            log.info("Initialization: Orthogonal")
            self._orthogonal_init()
        elif init_method.lower() == "kaiming":
            log.info("Initialization: Kaiming normal")
            self._kaiming_init()
        else:
            raise ValueError(init_method)
