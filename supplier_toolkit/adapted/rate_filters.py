"""Shared causal filters. Legacy 250/500 paths retained; explicit 1000 SPS design.
1000-SPS profile: 2nd-order Butterworth bandstop 48–52 Hz then 2nd-order
Butterworth highpass 70 Hz, matching the source EMG filter's intended kind.
These are new coefficients and require newly trained models.
"""
import numpy as np
from scipy.signal import butter
from legacy_filters import NotchFilter as LegacyNotch, EXGFilter as LegacyEXG
PROFILE = 'npg-native1000-bs48_52-hp70-v1'
SAMPLING_RATE = 1000


class CausalSOS:
    def __init__(self, sos):
        self.sos = np.asarray(sos, dtype=float)
        self.state = np.zeros((len(sos), 2))
    def process(self, value):
        value = float(value)
        for row, state in zip(self.sos, self.state):
            b0,b1,b2,a0,a1,a2 = row
            result = (b0*value+state[0])/a0
            state[0] = b1*value-a1*result+state[1]
            state[1] = b2*value-a2*result
            value = result
        return value


class NotchFilter:
    def __init__(self):
        self.legacy=LegacyNotch(); self.rate=None; self.filters={}
    def set_sampling_rate(self, rate):
        if rate not in (250,500,1000): raise ValueError('Unsupported sample rate')
        self.rate=rate;self.filters={}
        self.legacy.set_sampling_rate(rate)
    def process(self, value, kind):
        if kind not in (0,1,2): raise ValueError('Unsupported notch type')
        if self.rate is None: raise ValueError('Set sampling rate first')
        if self.rate!=1000:return self.legacy.process(value,kind)
        if not kind:return float(value)
        if kind not in self.filters:
            center=50 if kind==1 else 60
            self.filters[kind]=CausalSOS(butter(2,[center-2,center+2],btype='bandstop',fs=self.rate,output='sos'))
        return self.filters[kind].process(value)


class EXGFilter:
    def __init__(self):
        self.legacy=LegacyEXG();self.rate=None;self.filter=None
    def set_bits(self,bits,rate):
        if rate not in (250,500,1000):raise ValueError('Unsupported sample rate')
        self.rate=rate;self.legacy.set_bits(bits,rate)
        self.filter=CausalSOS(butter(2,70,btype='highpass',fs=rate,output='sos')) if rate==1000 else None
    def process(self,value,kind):
        if self.rate is None:raise ValueError('Set sampling rate first')
        if self.rate!=1000:return self.legacy.process(value,kind)
        if kind==0:return float(value)
        if kind!=4:raise ValueError('Only Off/EMG supported')
        return self.filter.process(value)
