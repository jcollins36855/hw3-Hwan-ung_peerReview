import numpy as np
from ligotools import readligo as rl

def test_dq_channel_to_seglist_all_zero():
	dq0 = np.zeros(8, dtype=int)
	assert rl.dq_channel_to_seglist(dq0, fs=1) == []

def test_seglist_single_one_fs2():
    dq = np.zeros(10, int)
    dq[5] = 1
    (s,) = rl.dq_channel_to_seglist(dq, fs=2) 
    assert (s.start, s.stop) == (10, 12)
