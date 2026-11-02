"""Every dot = 1 measurement of a hydrogen electron (4d, m=0). Points sampled exactly from |psi|^2. MP4 1080x1920."""
import math, subprocess, sys, numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "stix"

W, H, FPS, SECONDS = 1080, 1920, 30, 31
NF = FPS * SECONDS
PREVIEW = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else None
OUT = "every_dot_is_a_measurement.mp4"
import os; GF = os.environ.get("FONT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")) + "/"; BI = GF + "Poppins-BoldItalic.ttf"; MED = GF + "Poppins-Medium.ttf"
def ease(u): u = np.clip(u, 0.0, 1.0); return u * u * (3 - 2 * u)
def ef(u): return float(ease(u))

P = np.load("pts.npy"); XYZ = P[:, :3].astype(np.float64); SGN = P[:, 3]; NP = len(P)
GOLD = np.array([1.0, 0.70, 0.28]); BLUE = np.array([0.33, 0.62, 1.0])
COL = np.where(SGN[:, None] > 0, GOLD, BLUE)

# ---------------- timeline ----------------
T_COLLAPSE = (2.2, 2.75)
T_B0, T_B1, T_B2 = 2.9, 8.2, 15.0          # 1 dot -> 150 dots -> all dots
T_TILT = (15.5, 18.5)
T_CUT = (20.5, 22.5); T_UNCUT = (25.6, 26.6)
T_END = 26.8
def count(t):
    if t < T_B0: return 0
    if t < T_B1: return int(math.exp(math.log(150) * (t - T_B0) / (T_B1 - T_B0)))
    if t < T_B2: return int(math.exp(math.log(150) + (math.log(NP) - math.log(150)) * ef((t - T_B1) / (T_B2 - T_B1)) ** 1.1))
    return NP
def birth(i):  # first time dot i is shown (inverse of count)
    if i < 1: return T_B0
    if i < 150: return T_B0 + math.log(i + 1) / math.log(150) * (T_B1 - T_B0)
    return None

# ---------------- camera ----------------
SC = 15.5; CX, CY = 540, 1000; DIST = 160.0
def view(t):
    phi = 0.35 * t + 0.6 * ef((t - 15.2) / 4.0)                 # slow turn, a little faster at the reveal
    tilt = math.radians(14 + 22 * ef((t - T_TILT[0]) / (T_TILT[1] - T_TILT[0])) - 22 * ef((t - 23.5) / 2.5))
    return phi, tilt
def project(xyz, t):
    phi, tilt = view(t)
    c, s = math.cos(phi), math.sin(phi)
    x = xyz[:, 0] * c - xyz[:, 1] * s; y = xyz[:, 0] * s + xyz[:, 1] * c; z = xyz[:, 2]
    ct, st = math.cos(tilt), math.sin(tilt)
    yy = y * ct - z * st; zz = y * st + z * ct                 # yy = depth (toward camera is negative)
    k = DIST / (DIST + yy)
    return CX + x * SC * k, CY - zz * SC * k, yy, k

# ---------------- text ----------------
def text(runs, size):
    fs = [(t, ImageFont.truetype(p, size), c) for t, p, c in runs]; w = int(sum(f.getlength(t) for t, f, c in fs)) + 24
    im = Image.new("RGBA", (w, int(size * 1.45)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 12
    for t, f, c in fs: d.text((x, int(size * 1.1)), t, font=f, fill=c, anchor="ls"); x += f.getlength(t)
    return np.asarray(im).astype(np.float32) / 255
def mimg(sv, size):
    fig = plt.figure(figsize=(10, 2), dpi=100); fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, sv, fontsize=size, color="white", ha="center", va="center")
    fig.savefig("_m.png", transparent=True, dpi=100, bbox_inches="tight", pad_inches=0.06); plt.close(fig)
    return np.asarray(Image.open("_m.png").convert("RGBA")).astype(np.float32) / 255
def vstack(ims, gap=0):
    w = max(i.shape[1] for i in ims); h = sum(i.shape[0] for i in ims) + gap * (len(ims) - 1)
    out = np.zeros((h, w, 4), np.float32); y = 0
    for i in ims: x = (w - i.shape[1]) // 2; out[y:y + i.shape[0], x:x + i.shape[1]] = i; y += i.shape[0] + gap
    return out
def plate(im, pad=24, alpha=0.72):
    h, w = im.shape[:2]; P_ = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(P_)
    d.rounded_rectangle((0, 0, w + 2 * pad - 1, h + 2 * pad - 1), 26, fill=(0, 0, 0, int(255 * alpha)), outline=(255, 255, 255, 38), width=2)
    out = np.asarray(P_).astype(np.float32).copy() / 255; reg = out[pad:pad + h, pad:pad + w]; al = im[..., 3:4]
    reg[..., :3] = reg[..., :3] * (1 - al) + im[..., :3] * al; reg[..., 3:4] = np.maximum(reg[..., 3:4], al); return out
def paste(fr, im, x, y, a):
    if a <= 0.003: return
    h, w = im.shape[:2]; x, y = int(x - w / 2), int(y - h / 2)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = im[y0 - y:y1 - y, x0 - x:x1 - x]; al = s[..., 3:4] * a
    fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - al) + s[..., :3] * al
WH = (255, 255, 255, 255); GD = (255, 186, 80, 255); BL = (95, 165, 255, 255); GR = (165, 172, 190, 255)
HOOK = vstack([text([("Every time you look,", BI, WH)], 70), text([("the electron is", BI, WH)], 70), text([("somewhere else.", BI, GD)], 70)], -10)
TOP1 = vstack([text([("1 dot = ", BI, WH), ("1 measurement", BI, GD)], 68), text([("where the electron was found", MED, GR)], 32)])
TOP2 = vstack([text([("Random.", BI, WH)], 72), text([("Every single time.", BI, WH)], 72)], -8)
TOP3 = vstack([text([("Random.", BI, WH)], 72), text([("Until it isn't.", BI, GD)], 72)], -8)
TOP4 = vstack([text([("This is a ", BI, WH), ("real atom.", BI, GD)], 72), text([("hydrogen, 1 electron", MED, GR)], 32)], -4)
TOP5 = vstack([text([("Cut it open.", BI, WH)], 72)], 0)
FORM = plate(vstack([mimg(r"$P(\mathbf{r}) = |\psi_{4,2,0}(\mathbf{r})|^2$", 36),
                     text([("n = 4   l = 2   m = 0", MED, WH)], 32),
                     text([("gold = wave +", MED, GD), ("      ", MED, GD), ("blue = wave −", MED, BL)], 28)], 4))
NODE = plate(vstack([text([("a shell where the electron", MED, WH)], 34), text([("is ", MED, WH), ("never", BI, GD), (" found", MED, WH)], 36)]))
END = vstack([text([("No orbit. No path.", BI, WH)], 66), text([("Just probability.", BI, GD)], 66)], -6)
CNT_L = text([("measurements", MED, GR)], 30)
HANDLE = text([("@AlexCarten_", MED, (115, 115, 128, 255))], 26)
_d = {}
def num(n):
    s = f"{n:,}"
    if s not in _d: _d[s] = text([(s, BI, WH)], 70)
    return _d[s]

# ---------------- frame ----------------
def splat(acc, x, y, w):
    xi = np.floor(x).astype(int); yi = np.floor(y).astype(int); fx = x - xi; fy = y - yi
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        X_, Y_ = xi + dx, yi + dy; m = (X_ >= 0) & (X_ < W) & (Y_ >= 0) & (Y_ < H)
        for c in range(3): np.add.at(acc[..., c], (Y_[m], X_[m]), (w[:, c] * ww[:, None][:, 0][:, None])[m, 0] if False else w[m, c] * ww[m])
def frame(t):
    acc = np.zeros((H, W, 3), np.float32)
    n = count(t) if t >= T_B0 else NP
    collapse = 0.0
    if t < T_B0:
        collapse = ef((t - T_COLLAPSE[0]) / (T_COLLAPSE[1] - T_COLLAPSE[0]))
        if t >= T_COLLAPSE[1]: n = 0
    sx, sy, depth, k = project(XYZ[:max(n, 1)] * (1 - collapse) + 0.0, t)
    sx, sy, depth, k = sx[:n], sy[:n], depth[:n], k[:n]
    col = COL[:n].copy()
    # cut: hide the half nearest the camera
    cut = ef((t - T_CUT[0]) / (T_CUT[1] - T_CUT[0])) - ef((t - T_UNCUT[0]) / (T_UNCUT[1] - T_UNCUT[0]))
    vis = np.ones(n)
    if cut > 0 and n:
        plane = 60.0 * (1 - cut)                         # sweeps from far in front to the centre
        vis = np.clip((depth - (-plane)) / 1.5 + 0.0, 0, 1) if False else np.clip((depth + plane) / 2.0, 0, 1)
    if n:
        dens = n / NP
        gain = 0.62 / (dens ** 0.22 + 0.02)        # each dot bright when few, softer when many
        w = col * (gain * vis * (0.75 + 0.25 * k))[:, None]
        splat(acc, sx, sy, w.astype(np.float32))
    # tone map + glow
    soft = gaussian_filter(acc, (0.9, 0.9, 0))
    img = 1 - np.exp(-soft * 1.6)
    sm = img[::4, ::4]; gl = gaussian_filter(sm, (3, 3, 0)) * 0.9 + gaussian_filter(sm, (12, 12, 0)) * 0.9
    fr = np.clip(img + zoom(gl, (4, 4, 1), order=1)[:H, :W] * 0.8, 0, 1)
    # collapse flash at the nucleus
    fl = math.exp(-((t - T_COLLAPSE[1]) / 0.12) ** 2)
    if fl > 0.01:
        yy, xx = np.ogrid[:H, :W]; r2 = (xx - CX) ** 2 + (yy - CY) ** 2
        fr = np.clip(fr + fl * np.exp(-r2 / (2 * 40.0 ** 2))[..., None] * np.array([1.0, 0.9, 0.75]), 0, 1)
    # newest dots flash with a ring while they are still countable
    if T_B0 <= t < T_B1 + 0.5:
        n_now = count(t)
        for i in range(max(0, n_now - 6), n_now):
            b = birth(i)
            if b is None: continue
            age = t - b
            if age < 0 or age > 0.6: continue
            x_, y_, _, _ = project(XYZ[i:i + 1], t); x_, y_ = int(x_[0]), int(y_[0])
            rr = 8 + 60 * age; a_ = (1 - age / 0.6) ** 2
            yy, xx = np.ogrid[max(0, y_ - 80):min(H, y_ + 80), max(0, x_ - 80):min(W, x_ + 80)]
            d = np.sqrt((xx - x_) ** 2 + (yy - y_) ** 2)
            ring = np.exp(-((d - rr) / 2.5) ** 2) * a_ * 0.8 + np.exp(-(d / 5.0) ** 2) * a_ * 1.2
            sl = fr[max(0, y_ - 80):min(H, y_ + 80), max(0, x_ - 80):min(W, x_ + 80)]
            sl[:] = np.clip(sl + ring[..., None] * COL[i], 0, 1)
    # ---- text ----
    paste(fr, HOOK, 540, 260, 1.0 - ef((t - 2.3) / 0.4))
    paste(fr, TOP1, 540, 230, ef((t - 2.9) / 0.5) * (1 - ef((t - 8.3) / 0.4)))
    paste(fr, TOP2, 540, 230, ef((t - 8.7) / 0.4) * (1 - ef((t - 11.6) / 0.4)))
    paste(fr, TOP3, 540, 230, ef((t - 12.0) / 0.4) * (1 - ef((t - 15.2) / 0.4)))
    paste(fr, TOP4, 540, 230, ef((t - 15.6) / 0.5) * (1 - ef((t - 20.2) / 0.4)))
    paste(fr, TOP5, 540, 200, ef((t - 20.6) / 0.4) * (1 - ef((t - 25.8) / 0.4)))
    ca = ef((t - 2.9) / 0.5) * (1 - ef((t - 15.8) / 0.5))
    if ca > 0:
        paste(fr, num(count(t)), 540, 1660, ca); paste(fr, CNT_L, 540, 1730, ca)
    paste(fr, FORM, 540, 1690, ef((t - 16.2) / 0.6) * (1 - ef((t - 20.3) / 0.4)))
    paste(fr, NODE, 540, 1690, ef((t - 22.6) / 0.5) * (1 - ef((t - 25.8) / 0.4)))
    paste(fr, END, 540, 1680, ef((t - T_END) / 0.7))
    paste(fr, HANDLE, 540, 1872, 1.0)
    return fr

if PREVIEW is not None:
    for f in PREVIEW:
        Image.fromarray((frame(f / FPS) * 255).astype(np.uint8)).resize((360, 640), Image.LANCZOS).save(f"f_{f:03d}.png")
    sys.exit()

ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "mv.mp4"], stdin=subprocess.PIPE)
for f in range(NF):
    t = f / FPS; fr = frame(t)
    if t > SECONDS - 0.4: fr = fr * max(0.0, (SECONDS - t) / 0.4)
    ff.stdin.write((fr * 255).astype(np.uint8).tobytes())
    if f % 90 == 0: print("frame", f, flush=True)
ff.stdin.close(); ff.wait()

# ---------------- sound ----------------
SR = 44100; NS = int(SECONDS * SR); tt = np.arange(NS) / SR; rng = np.random.default_rng(9); mix = np.zeros(NS)
for fq, am in ((55.0, 0.5), (82.4, 0.35), (110.0, 0.25), (164.8, 0.12)):
    mix += am * np.sin(2 * np.pi * fq * tt + rng.random() * 6) * (0.7 + 0.3 * np.sin(2 * np.pi * tt / 6.5 + fq))
mix *= 0.08 * np.minimum(1, tt / 0.3) * np.minimum(1, (SECONDS - tt) / 1.5)
def add(sig, t0):
    i = int(t0 * SR); j = min(i + len(sig), NS); mix[i:j] += sig[:j - i]
def tone(fq, dur, dec, amp):
    n = int(dur * SR); x = np.arange(n) / SR
    return amp * (np.sin(2 * np.pi * fq * x) + 0.3 * np.sin(4 * np.pi * fq * x)) * np.exp(-x * dec) * np.minimum(1, x / 0.003)
w = np.convolve(rng.standard_normal(NS), np.ones(30) / 30, "same")
mix += 0.7 * w * np.exp(-((tt - 2.55) / 0.18) ** 2)                                  # implosion
add(tone(70.0, 1.5, 3.0, 0.5), T_COLLAPSE[1])
pent = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
for i in range(1, 150):                                                                # 1 tick per dot while countable
    b = birth(i)
    if b is None or (i > 25 and i % 3): continue
    add(tone(pent[i % 6] * (1 if i < 40 else 2), 0.25, 18, 0.10 if i < 25 else 0.05), b)
hiss = np.convolve(rng.standard_normal(NS), np.ones(6) / 6, "same")
mix += 0.35 * hiss * np.clip((tt - T_B1) / (T_B2 - T_B1), 0, 1) ** 2 * (tt < T_B2 + 0.05)   # rain of measurements
for fq, am in ((392.0, 0.25), (587.3, 0.18), (784.0, 0.12)): add(tone(fq, 3.5, 1.2, am), T_B2)
add(tone(98.0, 2.0, 2.0, 0.35), T_CUT[1]); add(tone(523.3, 3.0, 1.5, 0.15), T_END); add(tone(784.0, 3.0, 1.5, 0.1), T_END + 0.15)
mix = mix / np.abs(mix).max() * 0.85
sf.write("ma.wav", np.stack([mix, mix], 1).astype(np.float32), SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "mv.mp4", "-i", "ma.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], check=True)
print("done", OUT)
