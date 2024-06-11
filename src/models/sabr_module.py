# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional, Tuple

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

log = utils.get_pylogger(__name__)


class SABRModule(BaseBoundedModulePL):
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
        init_method: str = "ibp",
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

    def load_state_dict_from_checkpoint(self, path, strict=False):
        state_dict_sabr = torch.load(path)
        state_dict = {}
        for k, v in state_dict_sabr.items():
            # remove prefix 'blocks.'
            k_new = k[7:]
            state_dict[k_new] = v

        log.info(f"Checkpoint loaded from {path}")
        ret = self.load_state_dict(state_dict, strict=strict)
        return ret
