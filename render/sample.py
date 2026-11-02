import numpy as np
rng = np.random.default_rng(42)
def psi(x, y, z):
    r = np.sqrt(x*x+y*y+z*z) + 1e-9; ct = z / r
    return r**2 * (1 - r/12.0) * np.exp(-r/4.0) * (3*ct*ct - 1)          # hydrogen 4d, m = 0 (unnormalised)
L = 48.0; g = np.linspace(-L, L, 161); X, Y, Z = np.meshgrid(g, g, g)
pmax = (psi(X, Y, Z)**2).max() * 1.05
out = []; n = 0
while n < 130000:
    c = rng.uniform(-L, L, (4_000_000, 3)); p = psi(*c.T)**2 / pmax
    k = rng.random(len(c)) < p; s = c[k]; out.append(np.c_[s, np.sign(psi(*s.T))]); n += len(s)
P = np.concatenate(out)[:130000]; rng.shuffle(P)
np.save("pts.npy", P.astype(np.float32)); print(P.shape, np.abs(P[:, :3]).max(0), (P[:, 3] > 0).mean())
