"""object_qa reproduction: GRAFT + no intervention + RP2 + RP10, 3 seeds each = 12 runs."""
from sweeps.respmon_objinv_gr_canon_3seed import runs as _gr
from sweeps.respmon_ivainv_dn_rp_3seed import runs as _dnrp
from sweeps.respmon_rp_doses_canon_3seed import runs as _doses

OBJ = "configs/object_qa_respmon.yaml"
runs = (
    _gr
    + [r for r in _dnrp if r["config"] == OBJ]                       # donothing + rp2
    + [r for r in _doses if r["config"] == OBJ
       and r["reward_penalty_amount"] == 10.0]                       # rp10 (paper's pick)
)
assert len(runs) == 12, len(runs)
per_gpu = 3
no_baseline = True
