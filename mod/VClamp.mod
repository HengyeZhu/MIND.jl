COMMENT
NEURON's three-level two-electrode clamp, for secondorder=0. The three
linear amplifier equations are eliminated algebraically in clamp(), keeping
their previous-step states and after_cvode current update. This changes no
solver or language semantics for user MOD files.
ENDCOMMENT
DEFINE NSTEP 3
NEURON {
    POINT_PROCESS VClamp
    ELECTRODE_CURRENT i
    RANGE e0, vo0, vi0, dur, amp, gain, rstim, tau1, tau2, fac, i
}
UNITS { (nA) = (nanoamp) (mV) = (millivolt) (uS) = (microsiemens) }
PARAMETER {
    dur[NSTEP] (ms) <0, 1e9>
    amp[NSTEP] (mV)
    gain = 1e5 <0, 1e9>
    rstim = 1 (megohm) <1e-9, 1e9>
    tau1 = .001 (ms) <0, 1e9>
    tau2 = 0 (ms) <0, 1e9>
    e0 (mV) vo0 (mV) vi0 (mV)
    fac = 0 <1, 10>
}
ASSIGNED { v (mV) dt (ms) i (nA) stim (mV) tc (ms) }
STATE { e (mV) vo (mV) vi (mV) }
INITIAL { e0 = 0 vo = v vo0 = v vi = v vi0 = v e = 0 }
BREAKPOINT {
    SOLVE update METHOD after_cvode
    vstim()
    i = icur()
}
PROCEDURE vstim() {
    LOCAL j, selected
    tc = 0
    j = 0
    selected = 0
    while (j < NSTEP && !selected) {
        stim = amp[j]
        tc = tc + dur[j]
        if (t < tc) { tc = tc + 100 selected = 1 }
        j = j + 1
    }
}
PROCEDURE clamp() {
    LOCAL t1, t2
    t1 = tau1/dt
    t2 = tau2/dt
    e = ((1+t1)*((1-fac)*v+fac*t2*vo0/(1+t2))-stim-t1*(vi0-e0)) / (2+t1+(1+t1)*fac*gain/(1+t2))
    vo = (t2*vo0-gain*e)/(1+t2)
    vi = v+fac*vo-fac*v
}
FUNCTION icur() (nA) {
    if (t > tc) { e0 = 0 vo0 = 0 icur = 0 }
    else { clamp() icur = (vo-v)/rstim }
}
PROCEDURE update() { i = icur() e0 = e vo0 = vo vi0 = vi }
