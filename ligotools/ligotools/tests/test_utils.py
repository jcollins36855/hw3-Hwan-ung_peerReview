import numpy as np
import os
from ligotools.utils import whiten, write_wavfile, reqshift
from scipy.io import wavfile

def test_whiten_scales_with_unit_psd():
    fs = 1024
    dt = 1.0 / fs
    t = np.arange(0, 1, dt)
    x = np.sin(2*np.pi*50*t) + 0.1*np.random.RandomState(0).normal(size=t.size)

    interp_psd = lambda f: np.ones_like(f)

    y = whiten(x, interp_psd, dt)

    # utils.whiten normalization costnant = sqrt(2 * dt)
    expected = x * np.sqrt(2*dt)
    assert np.allclose(y, expected, rtol=1e-6, atol=1e-6)

def test_reqshift_moves_tone_frequency():
    fs = 4096
    t = np.arange(0, 1, 1/fs)
    f0 = 100
    sig = np.sin(2*np.pi*f0*t)

    shifted = reqshift(sig, fshift=50, sample_rate=fs)

    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    dom_f = freqs[np.argmax(np.abs(np.fft.rfft(shifted)))]
    assert abs(dom_f - (f0 + 50)) <= 2  