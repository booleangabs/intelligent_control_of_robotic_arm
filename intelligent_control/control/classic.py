import numpy as np

class PID:
    def __init__(self, out_range, kp, ki=0, kd=0, max_registry_length=10):
        self.max_out = out_range[1]
        self.min_out = out_range[0]
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_hist = []
        self.max_registry_length = max_registry_length

    def __call__(self, error):
        if len(self.error_hist) >= self.max_registry_length:
            self.error_hist.pop(0)
        self.error_hist.append(error)

        # Proportional
        u = self.kp * error

        # Derivative
        if self.kd > 0 and len(self.error_hist) > 1:
            derivative = self.error_hist[-1] - self.error_hist[-2]
            u += self.kd * derivative

        # Integral
        if self.ki > 0:
            integral = sum(self.error_hist)
            u += self.ki * integral

        return np.clip(u, self.min_out, self.max_out)
