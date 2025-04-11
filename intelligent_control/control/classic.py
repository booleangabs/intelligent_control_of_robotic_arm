import numpy as np

class PID:
    def __init__(self, kp, out_range, ki=0, kd=0, max_registry_length=10):
        self.max_out = out_range[1]
        self.min_out = out_range[0]
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_hist = []
        self.max_registry_length = max_registry_length
        self.integral = 0.0  # Separate integral state for anti-windup

    def __call__(self, error):
        if len(self.error_hist) >= self.max_registry_length:
            self.error_hist.pop(0)
        self.error_hist.append(error)

        # Proportional
        u_p = self.kp * error

        # Derivative
        u_d = 0.0
        if self.kd > 0 and len(self.error_hist) > 1:
            derivative = self.error_hist[-1] - self.error_hist[-2]
            u_d = self.kd * derivative

        # Integral with anti-windup
        u_i = 0.0
        if self.ki > 0:
            self.integral += error
            u_i = self.ki * self.integral

        # Total output before clamping
        u = u_p + u_i + u_d

        # Anti-windup: prevent integral accumulation if output is saturated
        u_clipped = np.clip(u, self.min_out, self.max_out)

        return u_clipped
