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


def brentq_solve(
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
    """Finding the batch of roots of the vectorized function `f` using the brentq method.
    Inspired by https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/brentq.c

    Args:
        f (Callable): function, which takes a batch of scalars as input,
                    and outputs a corresponding batch of scalars.
        bracket (Tuple[torch.Tensor, torch.Tensor]): lower and upper bounds for the roots.
        fbracket (Tuple[torch.Tensor, torch.Tensor], optional): function evaluations at the lower and upper bounds of the roots.
        args (Tuple, optional): additional arguments of the function `f`. Defaults to ().
        xtol (float, optional): absolute tolerance of the solution. Defaults to 2e-12.
        rtol (float, optional): relative tolerance of the solution. Defaults to 8.882e-16.
        maxiter (int, optional): maximum number of iterations. Defaults to 100.
        ftol (float, optional): function value tolerance of the solution. Defaults to 0.0.
        return_history (bool, optional): a flag whether or not to return the history of iterations.
                    Defaults to False.

    Returns:
        Tuple[torch.Tensor, RootResultsBatch]: return the batch of roots of the function `f`,
                    and the `RootResultsBatch` object.
                    If `return_history=True`, also return a dictionary with a history of iterations.
    """
    # xb is the current estimate
    # xa is the previous estimate
    # xc is the pre-previous estimate
    xa, xb = bracket
    xc = xa.detach().clone()
    dm = (xb - xa).detach().clone()
    dcur = dm.detach().clone()
    dpre = dm.detach().clone()
    dtry = dm.detach().clone()
    dfa = dm.detach().clone()
    dfc = dm.detach().clone()

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
    fc = fa.detach().clone()

    idx_signerr = fa * fb > 0
    res.inprogress[idx_signerr] = False
    res.flag[idx_signerr] = _ESIGNERR

    idx_converged = torch.abs(fa) <= ftol
    res.inprogress[idx_converged] = False
    res.flag[idx_converged] = _ECONVERGED
    res.root[idx_converged] = xa[idx_converged]

    idx_converged = torch.abs(fb) <= ftol
    res.inprogress[idx_converged] = False
    res.flag[idx_converged] = _ECONVERGED
    res.root[idx_converged] = xb[idx_converged]

    history = {"f": [], "x": []}
    for i in range(maxiter):
        if torch.all(~res.inprogress):
            break
        res.iterations[res.inprogress] += 1
        idx_inprogress = res.inprogress.type_as(fb)
        # if (fa != 0 && fb != 0 && (signbit(fa) != signbit(fb)))
        cond1 = torch.logical_and(idx_inprogress, fa * fb < 0)
        xc[cond1] = xa[cond1].detach().clone()
        fc[cond1] = fa[cond1].detach().clone()
        dcur[cond1] = (xb[cond1] - xa[cond1]).detach().clone()
        dpre[cond1] = (xb[cond1] - xa[cond1]).detach().clone()

        # if (fabs(fc) < fabs(fb))
        cond2 = torch.logical_and(idx_inprogress, torch.abs(fc) < torch.abs(fb))
        xa[cond2] = xb[cond2].detach().clone()
        xb[cond2] = xc[cond2].detach().clone()
        xc[cond2] = xa[cond2].detach().clone()
        fa[cond2] = fb[cond2].detach().clone()
        fb[cond2] = fc[cond2].detach().clone()
        fc[cond2] = fa[cond2].detach().clone()

        tol1 = (xtol + rtol * torch.abs(xb)) / 2
        dm = 0.5 * (xc - xb).detach().clone()
        idx_converged = torch.logical_and(
            idx_inprogress,
            torch.logical_or(torch.abs(fb) <= ftol, torch.abs(dm) < tol1),
        )
        res.inprogress[idx_converged] = False
        res.flag[idx_converged] = _ECONVERGED
        res.root[idx_converged] = xb[idx_converged]

        idx_inprogress = res.inprogress.type_as(fb)
        # if (fabs(dpre) > tol1 && fabs(fb) < fabs(fa))
        cond3 = torch.logical_and(torch.abs(dpre) > tol1, torch.abs(fb) < torch.abs(fa))
        # if (xa == xc)
        cond3_1 = xa == xc
        idx_lin = torch.logical_and(idx_inprogress, torch.logical_and(cond3, cond3_1))
        idx_quad = torch.logical_and(idx_inprogress, torch.logical_and(cond3, ~cond3_1))
        # linear approximation
        dtry[idx_lin] = (
            (-fb[idx_lin] * (xb[idx_lin] - xa[idx_lin]) / (fb[idx_lin] - fa[idx_lin]))
            .detach()
            .clone()
        )
        # inverse quadratic approximation
        dfa[idx_quad] = (fa[idx_quad] - fb[idx_quad]) / (
            xa[idx_quad] - xb[idx_quad]
        ).detach().clone()
        dfc[idx_quad] = (fc[idx_quad] - fb[idx_quad]) / (
            xc[idx_quad] - xb[idx_quad]
        ).detach().clone()
        dtry[idx_quad] = (
            (
                -fb[idx_quad]
                * (fc[idx_quad] * dfc[idx_quad] - fa[idx_quad] * dfa[idx_quad])
                / (dfc[idx_quad] * dfa[idx_quad] * (fc[idx_quad] - fa[idx_quad]))
            )
            .detach()
            .clone()
        )

        cond3_2 = 2 * torch.abs(dtry) < torch.minimum(
            torch.abs(dpre), 3 * torch.abs(dm) - tol1
        )
        idx_good = torch.logical_and(idx_inprogress, torch.logical_and(cond3, cond3_2))
        idx_bisect = torch.logical_and(
            idx_inprogress, torch.logical_and(cond3, ~cond3_2)
        )
        # good short step
        dpre[idx_good] = dcur[idx_good].detach().clone()
        dcur[idx_good] = dtry[idx_good].detach().clone()
        # bisect
        dpre[idx_bisect] = dm[idx_bisect].detach().clone()
        dcur[idx_bisect] = dm[idx_bisect].detach().clone()

        # else (not cond3) -> bisect
        idx_else = torch.logical_and(idx_inprogress, ~cond3)
        dpre[idx_else] = dm[idx_else].detach().clone()
        dcur[idx_else] = dm[idx_else].detach().clone()

        xa[idx_inprogress.bool()] = xb[idx_inprogress.bool()].detach().clone()
        fa[idx_inprogress.bool()] = fb[idx_inprogress.bool()].detach().clone()
        cond4 = torch.abs(dcur) > tol1
        idx_step = torch.logical_and(idx_inprogress, cond4)
        xb[idx_step] = (xb[idx_step] + dcur[idx_step]).detach().clone()
        cond4_1 = dm > 0
        idx_plus = torch.logical_and(idx_inprogress, torch.logical_and(~cond4, cond4_1))
        idx_minus = torch.logical_and(
            idx_inprogress, torch.logical_and(~cond4, ~cond4_1)
        )
        xb[idx_plus] = (xb[idx_plus] + tol1[idx_plus]).detach().clone()
        xb[idx_minus] = (xb[idx_minus] - tol1[idx_minus]).detach().clone()

        fb = f(xb, *args).detach().clone()
        res.function_calls[res.inprogress] += 1

        if return_history:
            history["f"].append(fb.detach().clone())
            history["x"].append(xb.detach().clone())

    res.root[res.inprogress] = xb[res.inprogress]
    res.flag[res.inprogress] = _ECONVERR
    res.converged = res.flag == _ECONVERGED
    res.convert_flag()
    roots = res.root.type_as(xb)
    return roots, res, history
