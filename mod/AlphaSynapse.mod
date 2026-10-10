NEURON {
    POINT_PROCESS AlphaSynapse
    RANGE onset, tau, gmax, e, i
    NONSPECIFIC_CURRENT i
}
UNITS { (nA) = (nanoamp) (mV) = (millivolt) (uS) = (microsiemens) }
PARAMETER {
    onset = 0 (ms)
    tau = 0.1 (ms) <1e-3, 1e6>
    gmax = 0 (uS) <0, 1e9>
    e = 0 (mV)
}
ASSIGNED { v (mV) i (nA) g (uS) }
BREAKPOINT {
    if (gmax) { at_time(onset) }
    g = gmax * alpha((t - onset)/tau)
    i = g * (v - e)
}
FUNCTION alpha(x) {
    if (x < 0 || x > 10) { alpha = 0 }
    else { alpha = x * exp(1 - x) }
}
