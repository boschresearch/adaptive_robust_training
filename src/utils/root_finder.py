# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import ABC, abstractmethod

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src import utils  # noqa: E402
from src.utils.solvers import bisect, brentq  # noqa: E402

log = utils.get_pylogger(__name__)


class RootFinderBase(ABC):
    def __init__(
        self, xtol: float = 2e-12, rtol: float = 8.882e-16, maxiter: int = 100
    ) -> None:
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter

    @abstractmethod
    def solve(self, f):
        pass


class PytorchOptimizeRootScalarBatch(RootFinderBase):
    def __init__(
        self,
        xtol: float = 2e-12,
        rtol: float = 8.882e-16,
        maxiter: int = 100,
        ftol: float = 0.0,
    ) -> None:
        super().__init__(xtol, rtol, maxiter)
        self.ftol = ftol
        self.history = {"f": [], "x": []}

    def solve(
        self,
        f,
        args=(),
        method="bisect",
        bracket=None,
        x0=None,
        x1=None,
        fprime=None,
        fprime2=None,
        options=None,
        return_history=False,
        fbracket=None,
    ):
        if method.lower() not in ("bisect", "brentq"):
            msg = f"Pytorch root finder does not support the method {method}"
            log.error(msg)
            raise ValueError(msg)
        if method.lower() == "bisect":
            roots, res, self.history = bisect.bisect_solve(
                f,
                bracket=bracket,
                fbracket=fbracket,
                args=args,
                xtol=self.xtol,
                rtol=self.rtol,
                maxiter=self.maxiter,
                ftol=self.ftol,
                return_history=return_history,
            )
        else:
            roots, res, self.history = brentq.brentq_solve(
                f,
                bracket=bracket,
                fbracket=fbracket,
                args=args,
                xtol=self.xtol,
                rtol=self.rtol,
                maxiter=self.maxiter,
                ftol=self.ftol,
                return_history=return_history,
            )
        return roots, res
