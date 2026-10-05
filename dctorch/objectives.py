"""
Robust (Student-t) data misfit for stock SimPEG >=0.25.

phi_d(m) = (nu+1) * sum(log1p(r_i^2 / nu)),  r = W (dpred - dobs)

(x2 the fork version: 0.25 dropped the 1/2 factor, so at nu -> inf this
matches L2DataMisfit's ||W r||^2.) TorchLBFGS picks up ``nu`` automatically
and runs the same functional in torch; the numpy call/deriv here keep the
rest of the ecosystem (directives, printers) consistent.
"""
import numpy as np
from simpeg.data_misfit import L2DataMisfit


class StudentTDataMisfit(L2DataMisfit):
    def __init__(self, nu=4.0, **kwargs):
        super().__init__(**kwargs)
        self.nu = float(nu)

    def __call__(self, m, f=None):
        r = self.W * self.residual(m, f=f)
        return float((self.nu + 1.0) * np.sum(np.log1p(r**2 / self.nu)))

    def deriv(self, m, f=None):
        r = self.W * self.residual(m, f=f)
        wt = 2.0 * (self.nu + 1.0) * r / (self.nu + r**2)
        return self.simulation.Jtvec(m, self.W.T * wt, f=f)

    def robust_chi2(self, m=None, r_w=None):
        """Reweighted chi^2 control: sum(w r^2)/sum(w), w = (nu+1)/(nu+r^2)."""
        if r_w is None:
            r_w = self.W * self.residual(m)
        w = (self.nu + 1.0) / (self.nu + r_w**2)
        return float(np.sum(w * r_w**2) / np.sum(w))
