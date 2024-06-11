# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Callable, Optional, Tuple

import pyrootutils
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src.utils.solvers.common import (  # noqa: E402
    _ECONVERGED,
    _ECONVERR,
    _EINPROGRESS,
    _ESIGNERR,
    RootResultsBatch,
)


def bisect_solve(
    f: Callable,
    bracket: Tuple[torch.Tensor, torch.Tensor],
    fbracket: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    args: Tuple = (),
    xtol: float = 2e-12,
    rtol: float = 8.882e-16,
    maxiter: int = 100,
    ftol: float = 0.0,
    return_history: bool = False,
) -> Tuple[torch.Tensor, RootResultsBatch]:
    """Finding the batch of roots of the batch-vector-function using the bisection method.
    Inspired by https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/bisect.c

    Args:
        f (Callable): function, which takes a batch of scalars as input,
                    and outputs a corresponding batch of scalars.
        bracket (Tuple[torch.Tensor, torch.Tensor]): lower and upper bounds for the roots.
        fbracket (Tuple[torch.Tensor, torch.Tensor], optional): function evaluations at the lower and upper bounds of the roots.
        args (Tuple, optional): additional arguments of the function `f`. Defaults to ().
        xtol (float, optional): absolute tolerance of the solution. Defaults to 2e-12.
        rtol (float, optional): relative tolerance of the solution. Defaults to 8.882e-16.
        maxiter (int, optional): maximum number of iterations of the bisection. Defaults to 100.
        ftol (float, optional): function value tolerance of the solution. Defaults to 0.0.
        return_history (bool, optional): a flag whether or not to return the history of iterations.
                    Defaults to False.

    Returns:
        Tuple[torch.Tensor, RootResultsBatch]: return the batch of roots of the function `f`,
                    and the `RootResultsBatch` object.
                    If `return_history=True`, also return a dictionary with a history of iterations.
    """
    xa, xb = bracket
    N = xa.size(0)
    res = RootResultsBatch(N, examplar_tensor=xb)
    res.converged *= False
    res.inprogress += True
    res.flag += _EINPROGRESS

    if fbracket is not None:
        fa, fb = fbracket
    else:
        fa = f(xa, *args)
        fb = f(xb, *args)
        res.function_calls += 2

    idx_failed = fa * fb > 0
    res.inprogress[idx_failed] = False
    res.flag[idx_failed] = _ESIGNERR

    idx_con = torch.abs(fa) <= ftol
    res.inprogress[idx_con] = False
    res.flag[idx_con] = _ECONVERGED
    res.root[idx_con] = xa[idx_con]

    idx_con = torch.abs(fb) <= ftol
    res.inprogress[idx_con] = False
    res.flag[idx_con] = _ECONVERGED
    res.root[idx_con] = xb[idx_con]

    dm = (xb - xa).detach().clone()
    history = {"f": [], "x": []}
    for i in range(maxiter):
        if torch.all(~res.inprogress):
            break
        res.iterations[res.inprogress] += 1
        dm *= 0.5
        xm = (xa + dm).detach().clone()
        fm = f(xm, *args)
        if return_history:
            history["f"].append(fm.detach().clone())
            history["x"].append(xm.detach().clone())
        res.function_calls[res.inprogress] += 1
        xa[fm * fa >= 0] = xm[fm * fa >= 0].detach().clone()
        idx_con = torch.logical_and(
            res.inprogress.type_as(fm),
            torch.logical_or(
                torch.abs(fm) <= ftol, torch.abs(dm) < xtol + rtol * torch.abs(xm)
            ),
        )
        res.inprogress[idx_con] = False
        res.flag[idx_con] = _ECONVERGED
        res.root[idx_con] = xm[idx_con]
    res.root[res.inprogress] = xa[res.inprogress]
    res.flag[res.inprogress] = _ECONVERR
    res.converged = res.flag == _ECONVERGED
    res.convert_flag()
    roots = res.root.type_as(xb)
    return roots, res, history
