import numpy as np, soundfile as sf
from scipy.signal import fftconvolve
SR = 44100; SEC = 31.0; N = int(SR * SEC); t = np.arange(N) / SR; rng = np.random.default_rng(11)
def note(f): return 440.0 * 2 ** ((f - 69) / 12)
# chords (midi), 4 bars of ~7.75 s: Cmaj9 - Am9 - Fmaj7(#11) - Gsus -> resolve
CH = [[48, 55, 59, 62, 64], [45, 52, 55, 59, 60], [41, 48, 52, 57, 59], [43, 50, 55, 57, 62]]
BAR = SEC / 4
pad = np.zeros(N)
for b, ch in enumerate(CH):
    t0 = b * BAR; seg = (t >= t0 - 1.5) & (t < t0 + BAR + 2.5)
    env = np.clip((t - (t0 - 1.5)) / 2.5, 0, 1) * np.clip((t0 + BAR + 2.5 - t) / 3.0, 0, 1)
    for m in ch:
        f = note(m)
        for det in (-0.12, 0.0, 0.12):                       # soft chorus
            ph = rng.random() * 6
            pad += env * (np.sin(2 * np.pi * f * (1 + det / 100) * t + ph) + 0.18 * np.sin(4 * np.pi * f * t + ph)) / 3
pad *= 0.045 * (0.85 + 0.15 * np.sin(2 * np.pi * t / 9.0))
# swell when the atom forms (15 s) and at the end
swell = 1 + 0.6 * np.exp(-((t - 15.5) / 2.5) ** 2) + 0.3 * np.clip((t - 26.5) / 2, 0, 1)
pad *= swell
# sparse soft "glass" piano notes from the chord tones
pl = np.zeros(N)
def pluck(f, t0, amp):
    i = int(t0 * SR); n = min(int(4.0 * SR), N - i)
    if n <= 0: return
    x = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * x) + 0.35 * np.sin(4 * np.pi * f * x) * np.exp(-x * 3) + 0.1 * np.sin(6 * np.pi * f * x) * np.exp(-x * 6))
    pl[i:i + n] += amp * s * np.exp(-x * 1.4) * np.minimum(1, x / 0.004)
times = np.arange(0.4, SEC - 2, 1.55)
for k, tp in enumerate(times):
    ch = CH[min(int(tp / BAR), 3)]
    pluck(note(ch[(k * 3) % len(ch)] + 24), tp + rng.uniform(-0.05, 0.05), 0.05 if 3 < tp < 15 else 0.035)
# warm sub
sub = 0.05 * np.sin(2 * np.pi * note(CH[0][0] - 12) * t) * np.clip(t / 3, 0, 1)
for b, ch in enumerate(CH):
    m = (t >= b * BAR) & (t < (b + 1) * BAR); sub[m] = 0.05 * np.sin(2 * np.pi * note(ch[0] - 12) * t[m])
mus = pad + pl + sub * 0.6
# reverb: exponentially decaying noise impulse, stereo
L = int(3.2 * SR); x = np.arange(L) / SR
irL = rng.standard_normal(L) * np.exp(-x * 1.6); irR = rng.standard_normal(L) * np.exp(-x * 1.6)
wetL = fftconvolve(mus, irL)[:N]; wetR = fftconvolve(mus, irR)[:N]
wl = wetL / np.abs(wetL).max(); wr = wetR / np.abs(wetR).max(); dry = mus / np.abs(mus).max()
st = np.stack([0.55 * dry + 0.45 * wl, 0.55 * dry + 0.45 * wr], 1)
fade = np.minimum(1, t / 1.0) * np.minimum(1, (SEC - t) / 2.0); st *= fade[:, None]
st = st / np.abs(st).max() * 0.8
sf.write("every_dot_music.wav", st.astype(np.float32), SR)
fx, sr2 = sf.read("ma.wav"); n = min(len(fx), N)
mix = st[:n] * 0.85 + fx[:n] * 0.55
mix = mix / np.abs(mix).max() * 0.85
sf.write("every_dot_music_and_sound.wav", mix.astype(np.float32), SR)
print("ok")
