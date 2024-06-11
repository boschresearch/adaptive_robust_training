# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


class EpsScheduler(object):
    def __init__(
        self,
        schedule_start=2,
        schedule_length=80,
        max_eps=0.4,
    ):
        self.schedule_start = schedule_start
        self.schedule_length = schedule_length
        self.max_eps = max_eps
        self.epoch_length = None

    def __repr__(self):
        return f"<EpsScheduler: max_eps {self.max_eps}, schedule_start {self.schedule_start}, schedule_length {self.schedule_length}>"

    def set_epoch_length(self, epoch_length):
        self.epoch_length = epoch_length

    def get_eps(self, step):
        if self.epoch_length is None:
            raise ValueError("Set epoch length first!")

        # Smooth schedule that slowly morphs into a linear schedule.
        mid_point = 0.25
        beta = 4.0
        init_value = 0.0
        final_value = self.max_eps
        # Batch number for schedule start
        init_step = (self.schedule_start - 1) * self.epoch_length
        # Batch number for schedule end
        final_step = (
            self.schedule_start + self.schedule_length - 1
        ) * self.epoch_length
        # Batch number for switching from exponential to linear schedule
        mid_step = int((final_step - init_step) * mid_point) + init_step
        t = (mid_step - init_step) ** (beta - 1.0)
        # find coefficient for exponential growth, such that at mid point the gradient is the same as a linear ramp to final value
        alpha = (final_value - init_value) / (
            (final_step - mid_step) * beta * t + (mid_step - init_step) * t
        )
        # value at switching point
        mid_value = init_value + alpha * (mid_step - init_step) ** beta
        # return init_value when we have not started
        is_ramp = float(step > init_step)
        # linear schedule after mid step
        is_linear = float(step >= mid_step)
        exp_value = init_value + alpha * float(step - init_step) ** beta
        linear_value = min(
            mid_value
            + (final_value - mid_value) * (step - mid_step) / (final_step - mid_step),
            final_value,
        )
        return (
            is_ramp * ((1.0 - is_linear) * exp_value + is_linear * linear_value)
            + (1.0 - is_ramp) * init_value
        )
