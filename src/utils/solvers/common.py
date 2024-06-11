# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch

_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_EINPROGRESS = 1

CONVERGED = "converged"
SIGNERR = "sign error"
CONVERR = "convergence error"
VALUEERR = "value error"
INPROGRESS = "in progress"


flag_map = {
    _ECONVERGED: CONVERGED,
    _ESIGNERR: SIGNERR,
    _ECONVERR: CONVERR,
    _EVALUEERR: VALUEERR,
    _EINPROGRESS: INPROGRESS,
}


class RootResultsBatch:
    """Represents the root finding results in a batch (inspired by `scipy.optimize.RootResults`)."""

    def __init__(self, N, examplar_tensor):
        self.N = N
        self.inprogress = torch.zeros(N).bool()
        self.root = torch.zeros(N).type_as(examplar_tensor)
        self.iterations = torch.zeros(N)
        self.function_calls = torch.zeros(N)
        self.converged = torch.zeros(N).bool()
        self.flag = torch.zeros(N)

    def __repr__(self):
        res = "Results of a root finding algorithm in a batch-wise manner"
        for i in range(self.N):
            attrs = ["inprogress", "converged", "function_calls", "iterations", "root"]
            m = max(map(len, attrs)) + 1
            res += f"\nElement {i}\n"
            res += "\n".join(
                [a.rjust(m) + ": " + repr(getattr(self, a)[i].item()) for a in attrs]
            )
            res += "\n" + "flag".rjust(m) + ": " + f"{self.flag[i]}"
        return res

    def convert_flag(self):
        str_flag = []
        for f in self.flag:
            str_flag.append(flag_map[f.item()])
        self.flag = str_flag
