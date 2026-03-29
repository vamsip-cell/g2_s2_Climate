# ╔══════════════════════════════════════════════════════════════════════╗
# ║   MILESTONE 4 — ENSEMBLE NOWCASTING + LAPLACE MOTION PERTURB         ║
# ╚══════════════════════════════════════════════════════════════════════╝
# ============================================================
# 1. INSTALL & IMPORTS
# ============================================================
!pip install pysteps matplotlib numpy scipy -q

import os, urllib.request, zipfile, glob, gzip, shutil, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================
# 2. DOWNLOAD DATASET
# ============================================================
data_path = "/content/pysteps_data"

if not os.path.exists(data_path):
    print("Downloading dataset...")
    url      = "https://github.com/pySTEPS/pysteps-data/archive/refs/heads/master.zip"
    zip_path = "/content/data.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("/content/")
    os.rename("/content/pysteps-data-master", data_path)
    print("Download complete.")
else:
    print("Dataset already exists.")

# ============================================================
# 3. FIND & DECOMPRESS .pgm.gz FILES
# ============================================================
gz_files = sorted(glob.glob(f"{data_path}/**/*.pgm.gz", recursive=True))
print(f"Found {len(gz_files)} .pgm.gz files")

N_STEPS  = 6          # forecast lead times  (test length)
# We need: training frames + 2 context frames + N_STEPS future frames
# minimum total = 3 (train) + 2 (context overlap) + N_STEPS = 11
N_FRAMES = min(30, len(gz_files))
if N_FRAMES < (N_STEPS + 5):
    raise Exception(f"Need >= {N_STEPS+5} files, found {len(gz_files)}")
print(f"Using {N_FRAMES} frames total")

decompressed_dir = "/tmp/pgm_decomp"
os.makedirs(decompressed_dir, exist_ok=True)

def decompress_pgm_gz(gz_path, out_dir):
    bn       = os.path.basename(gz_path).replace(".pgm.gz", ".pgm")
    out_path = os.path.join(out_dir, bn)
    if not os.path.exists(out_path):
        with gzip.open(gz_path, 'rb') as fi, open(out_path, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    return out_path

selected_files = [decompress_pgm_gz(f, decompressed_dir) for f in gz_files[:N_FRAMES]]
print(f"Decompressed {len(selected_files)} files.")

# ============================================================
# 4. LOAD RADAR DATA
# ============================================================
from pysteps.io.importers import import_fmi_pgm
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation

R_list, metadata = [], None
for f in selected_files:
    result = import_fmi_pgm(f)
    if   len(result) == 3: R_frame, _, metadata = result
    elif len(result) == 2: R_frame,    metadata = result
    else: raise Exception("Unexpected importer format")
    R_list.append(R_frame)

R = np.array(R_list)
print(f"Radar stack shape: {R.shape}  (frames × H × W)")

# ============================================================
# 5. dBZ → RAIN RATE → LOG TRANSFORM
# ============================================================
result = transformation.dB_transform(R, metadata=metadata, inverse=True)
R, metadata = result if isinstance(result, tuple) else (result, metadata)
R     = np.clip(np.nan_to_num(R, nan=0.0), 0, None)
R_log = np.log(R + 0.1)
print(f"R_log range: [{R_log.min():.3f}, {R_log.max():.3f}]")

# ============================================================
# 6. TRAIN / TEST SPLIT  ← KEY FIX
# ============================================================
# Layout of R_log (total N_FRAMES):
#
#   |<──────── TRAIN ──────────>|← ctx →|<──── TEST (future) ────>|
#   0                      TRAIN_END-3  TRAIN_END   TRAIN_END+N_STEPS
#
# TRAIN  → AR(2) params + motion estimation
# ctx    → last 2 training frames used as AR(2) context (t-1, t)
# TEST   → NEVER seen by the model; used only for verification
#
TRAIN_END = N_FRAMES - N_STEPS      # index of first test frame

# Training frames (includes the 2 context frames at the tail)
R_train   = R_log[:TRAIN_END]       # shape: (TRAIN_END, H, W)

# The 2 context frames fed into the forecasting model
R_input_t1 = R_log[TRAIN_END - 2]  # t-1  (2nd-to-last train frame)
R_input_t  = R_log[TRAIN_END - 1]  # t    (last train frame = "now")

# Ground-truth future frames for evaluation  ← replaces obs = R_log[-1]
future_obs  = R_log[TRAIN_END : TRAIN_END + N_STEPS]   # (N_STEPS, H, W)

print(f"\nTrain/Test split:")
print(f"  Training frames : 0 … {TRAIN_END-1}  ({TRAIN_END} frames)")
print(f"  Model input     : t-1={TRAIN_END-2},  t={TRAIN_END-1}")
print(f"  Test frames     : {TRAIN_END} … {TRAIN_END+N_STEPS-1}  ({N_STEPS} frames)")
print(f"  future_obs shape: {future_obs.shape}")

# ============================================================
# 7. MOTION ESTIMATION  (on last 3 TRAINING frames only)
# ============================================================
print("\nEstimating motion field from training data...")
V = dense_lucaskanade(R_train[-3:])
print(f"Motion field shape: {V.shape}")

# ============================================================
# 8. CASCADE DECOMPOSITION  (8 spatial scales via Gaussian)
# ============================================================
N_SCALES = 8

def cascade_decompose(field, n_scales=N_SCALES):
    scales, current = [], field.copy()
    for k in range(n_scales - 1):
        smoothed = gaussian_filter(current, sigma=2**k)
        scales.append(current - smoothed)
        current  = smoothed
    scales.append(current)
    return scales

def cascade_reconstruct(scales):
    return np.sum(scales, axis=0)

# Decompose the last observed frame for visualisation
obs_scales = cascade_decompose(R_input_t)
print(f"\nDecomposed into {N_SCALES} cascade scales.")
print(f"  Scale energy: {[f'{np.var(s):.4f}' for s in obs_scales]}")

# ============================================================
# 9. LAGRANGIAN ADVECTION HELPERS
# ============================================================
def warp_field(field, V, steps=1):
    H, W   = field.shape
    gy, gx = np.mgrid[0:H, 0:W].astype(float)
    src_x  = np.clip(gx - V[0]*steps, 0, W-1)
    src_y  = np.clip(gy - V[1]*steps, 0, H-1)
    x0 = np.floor(src_x).astype(int);  x1 = np.clip(x0+1, 0, W-1)
    y0 = np.floor(src_y).astype(int);  y1 = np.clip(y0+1, 0, H-1)
    wx = src_x - x0;  wy = src_y - y0
    return ((1-wy)*(1-wx)*field[y0,x0] + (1-wy)*wx*field[y0,x1] +
               wy *(1-wx)*field[y1,x0] +    wy *wx*field[y1,x1])

def lagrangian_to_eulerian(lag_field, V, steps=1):
    return warp_field(lag_field,  V, steps)

def eulerian_to_lagrangian(field, V, steps=1):
    return warp_field(field, -V, steps)

# ============================================================
# 10. LAPLACE MOTION PERTURBATION
# ============================================================
def estimate_motion_uncertainty(V):
    b_x = float(np.mean(np.abs(V[0] - np.mean(V[0]))))
    b_y = float(np.mean(np.abs(V[1] - np.mean(V[1]))))
    b_x = max(b_x, 0.3)
    b_y = max(b_y, 0.3)
    return b_x, b_y

def perturb_motion_laplace(V, b_x, b_y, smooth_sigma=4.0, decay=1.0):
    H, W = V.shape[1], V.shape[2]
    noise_x = np.random.laplace(loc=0, scale=b_x, size=(H, W))
    noise_y = np.random.laplace(loc=0, scale=b_y, size=(H, W))
    noise_x = gaussian_filter(noise_x, sigma=smooth_sigma)
    noise_y = gaussian_filter(noise_y, sigma=smooth_sigma)
    noise_x = noise_x / (noise_x.std() + 1e-9) * b_x
    noise_y = noise_y / (noise_y.std() + 1e-9) * b_y
    V_p    = V.copy().astype(float)
    V_p[0] = V[0] + decay * noise_x
    V_p[1] = V[1] + decay * noise_y
    return V_p

B_X, B_Y = estimate_motion_uncertainty(V)
print(f"\nLaplace motion perturbation scales → b_x={B_X:.3f}  b_y={B_Y:.3f} px/step")

# ============================================================
# 11. AR(2) PARAMETER ESTIMATION PER SCALE  (training frames only)
# ============================================================
def estimate_ar2_per_scale(R_log_train, n_scales=N_SCALES, V=None):
    """
    Estimate AR(2) phi1, phi2, sigma from TRAINING frames only.
    sigma is the mean residual std → calibrated stochastic noise.
    """
    n, params = len(R_log_train), []
    for j in range(n_scales):
        rho1_list, rho2_list, res_list = [], [], []
        for i in range(2, n):
            s_t  = cascade_decompose(R_log_train[i  ])[j]
            s_t1 = cascade_decompose(R_log_train[i-1])[j]
            s_t2 = cascade_decompose(R_log_train[i-2])[j]
            if V is not None:
                s_t  = eulerian_to_lagrangian(s_t,  V, steps=0)
                s_t1 = eulerian_to_lagrangian(s_t1, V, steps=1)
                s_t2 = eulerian_to_lagrangian(s_t2, V, steps=2)
            f0, f1, f2 = s_t.flatten(), s_t1.flatten(), s_t2.flatten()
            r1 = np.corrcoef(f0, f1)[0,1]
            r2 = np.corrcoef(f0, f2)[0,1]
            if np.isfinite(r1) and np.isfinite(r2):
                rho1_list.append(r1); rho2_list.append(r2)
                denom = 1 - r1**2 + 1e-6
                ph1   = r1*(1-r2)/denom
                ph2   = (r2-r1**2)/denom
                residual = f0 - (ph1*f1 + ph2*f2)
                res_list.append(np.std(residual))

        rho1  = float(np.mean(rho1_list)) if rho1_list else 0.9
        rho2  = float(np.mean(rho2_list)) if rho2_list else 0.8
        denom = 1 - rho1**2 + 1e-6
        phi1  = float(np.clip(rho1*(1-rho2)/denom,  -1.5, 1.5))
        phi2  = float(np.clip((rho2-rho1**2)/denom, -1.0, 1.0))
        # sigma from actual training residuals → calibrated to real variability
        sigma = float(np.mean(res_list)) if res_list else 0.01
        params.append((phi1, phi2, sigma))
        print(f"  Scale {j+1:2d}: phi1={phi1:.3f}  phi2={phi2:.3f}  sigma={sigma:.4f}")
    return params

print("\nEstimating AR(2) parameters on TRAINING frames only...")
scale_params = estimate_ar2_per_scale(R_train, N_SCALES, V)

# ============================================================
# 12. SINGLE-STEP STOCHASTIC CASCADE FORECAST
# ============================================================
def forecast_one_step(prev_scales_t, prev_scales_t1, scale_params,
                      V_use, step_idx, stochastic=True):
    forecast_scales = []
    for j, (phi1, phi2, sigma) in enumerate(scale_params):
        lag_t  = eulerian_to_lagrangian(prev_scales_t[j],  V_use, steps=step_idx)
        lag_t1 = eulerian_to_lagrangian(prev_scales_t1[j], V_use, steps=step_idx+1)
        R_next_lag = phi1 * lag_t + phi2 * lag_t1
        if stochastic and sigma > 0:
            R_next_lag += np.random.normal(0, sigma, R_next_lag.shape)
        forecast_scales.append(lagrangian_to_eulerian(R_next_lag, V_use, steps=step_idx+1))
    return forecast_scales

# ============================================================
# 13. DETERMINISTIC BASELINE
# ============================================================
def deterministic_nowcast(R_input_t, R_input_t1, scale_params, V, n_steps=6):
    """
    Uses R_input_t (last train frame) and R_input_t1 (second-to-last)
    as AR(2) context — no future data involved.
    """
    forecasts = []
    scales_t1 = cascade_decompose(R_input_t1)
    scales_t  = cascade_decompose(R_input_t)
    for k in range(n_steps):
        scales_next = forecast_one_step(scales_t, scales_t1, scale_params,
                                        V, step_idx=k, stochastic=False)
        forecasts.append(cascade_reconstruct(scales_next))
        scales_t1, scales_t = scales_t, scales_next
    return np.array(forecasts)

# ============================================================
# 14. ENSEMBLE NOWCASTING  (with Laplace motion perturbation)
# ============================================================
def ensemble_nowcast(R_input_t, R_input_t1, scale_params, V, b_x, b_y,
                     n_ens=30, n_steps=6,
                     motion_smooth_sigma=4.0,
                     perturb_motion=True):
    """
    Stochastic ensemble nowcast with:
      (a) AR(2) Gaussian innovation noise  (calibrated sigma from training)
      (b) Laplace motion perturbation      (advection uncertainty)

    Both context frames come from the TRAINING set tail — no leakage.
    Evaluated against future_obs which is entirely held-out.
    """
    all_members = []
    for m in range(n_ens):
        forecasts = []
        scales_t1 = cascade_decompose(R_input_t1)
        scales_t  = cascade_decompose(R_input_t)
        for k in range(n_steps):
            if perturb_motion:
                decay  = np.exp(-0.15 * k)
                V_use  = perturb_motion_laplace(V, b_x, b_y,
                                                smooth_sigma=motion_smooth_sigma,
                                                decay=decay)
            else:
                V_use  = V
            scales_next = forecast_one_step(scales_t, scales_t1, scale_params,
                                            V_use, step_idx=k, stochastic=True)
            forecasts.append(cascade_reconstruct(scales_next))
            scales_t1, scales_t = scales_t, scales_next
        all_members.append(forecasts)
    return np.array(all_members)   # (n_ens, n_steps, H, W)

# ============================================================
# 15. RUN FORECASTS
# ============================================================
N_ENS = 30

print(f"\nRunning deterministic nowcast ({N_STEPS} steps)...")
det_fc = deterministic_nowcast(R_input_t, R_input_t1, scale_params, V,
                                n_steps=N_STEPS)

print(f"Running ensemble WITHOUT motion perturbation ({N_ENS}×{N_STEPS})...")
ens_no_mp = ensemble_nowcast(R_input_t, R_input_t1, scale_params, V, B_X, B_Y,
                              n_ens=N_ENS, n_steps=N_STEPS,
                              perturb_motion=False)

print(f"Running ensemble WITH Laplace motion perturbation ({N_ENS}×{N_STEPS})...")
ens_mp    = ensemble_nowcast(R_input_t, R_input_t1, scale_params, V, B_X, B_Y,
                              n_ens=N_ENS, n_steps=N_STEPS,
                              perturb_motion=True)

mean_no_mp = np.mean(ens_no_mp, axis=0)
std_no_mp  = np.std( ens_no_mp, axis=0)
mean_mp    = np.mean(ens_mp,    axis=0)
std_mp     = np.std( ens_mp,    axis=0)
med_mp     = np.median(ens_mp,  axis=0)

print("Done running forecasts.")

# ============================================================
# 16. EVALUATION  — verified against FUTURE FRAMES (held-out test set)
# ============================================================
# future_obs[k] is the ground truth at lead-time k+1
# It was NEVER used for training or as model input  ← KEY FIX

def mse(a,b):  return float(np.mean((a-b)**2))
def mae(a,b):  return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(mse(a,b)))
def bias(a,b): return float(np.mean(a-b))
def ss(ens_step, obs_step):
    skill  = rmse(np.mean(ens_step, axis=0), obs_step)
    spread = float(np.mean(np.std(ens_step, axis=0)))
    return spread / (skill + 1e-9)

# Per-lead-time metrics vs the correct future observation
rmse_det  = [rmse(det_fc[k],      future_obs[k]) for k in range(N_STEPS)]
rmse_nmp  = [rmse(mean_no_mp[k],  future_obs[k]) for k in range(N_STEPS)]
rmse_mp_l = [rmse(mean_mp[k],     future_obs[k]) for k in range(N_STEPS)]
# ============================================================
# GRAPH 1 — RMSE Comparison
# ============================================================
steps = np.arange(1, N_STEPS+1)
plt.figure(figsize=(8,5))
plt.plot(steps, rmse_det, 'o-', label='Deterministic')
plt.plot(steps, rmse_nmp, 's--', label='Ensemble (no MP)')
plt.plot(steps, rmse_mp_l, 'd-', label='Ensemble (+LapMP)')
plt.xlabel("Lead Time (steps)")
plt.ylabel("RMSE")
plt.title("RMSE Comparison: Deterministic vs Ensemble")
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# GRAPH 2 — MAE Comparison
# ============================================================
mae_det = [mae(det_fc[k], future_obs[k]) for k in range(N_STEPS)]
mae_nmp = [mae(mean_no_mp[k], future_obs[k]) for k in range(N_STEPS)]
mae_mp = [mae(mean_mp[k], future_obs[k]) for k in range(N_STEPS)]
plt.figure(figsize=(8,5))
plt.plot(steps, mae_det, 'o-', label='Deterministic')
plt.plot(steps, mae_nmp, 's--', label='Ensemble (no MP)')
plt.plot(steps, mae_mp, 'd-', label='Ensemble (+LapMP)')
plt.xlabel("Lead Time (steps)")
plt.ylabel("MAE")
plt.title("MAE Comparison")
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# GRAPH 3 — MSE Comparison
# ============================================================
mse_det = [mse(det_fc[k], future_obs[k]) for k in range(N_STEPS)]
mse_nmp = [mse(mean_no_mp[k], future_obs[k]) for k in range(N_STEPS)]
mse_mp = [mse(mean_mp[k], future_obs[k]) for k in range(N_STEPS)]
plt.figure(figsize=(8,5))
plt.plot(steps, mse_det, 'o-', label='Deterministic')
plt.plot(steps, mse_nmp, 's--', label='Ensemble (no MP)')
plt.plot(steps, mse_mp, 'd-', label='Ensemble (+LapMP)')
plt.xlabel("Lead Time (steps)")
plt.ylabel("MSE")
plt.title("MSE Comparison")
plt.legend()
plt.grid(True)
plt.show()
# ============================================================
# GRAPH 4 — Step-1 Comparison (Bar Chart)
# ============================================================
labels = ['Deterministic', 'Ens (no MP)', 'Ens (+MP)']
rmse_vals = [
rmse(det_fc[0], future_obs[0]),
rmse(mean_no_mp[0], future_obs[0]),
rmse(mean_mp[0], future_obs[0])
]
plt.figure(figsize=(6,4))
plt.bar(labels, rmse_vals)
plt.ylabel("RMSE")
plt.title("Step-1 RMSE Comparison")
plt.show()
ss_no_mp = [ss(ens_no_mp[:,k], future_obs[k]) for k in range(N_STEPS)]
ss_mp = [ss(ens_mp[:,k], future_obs[k]) for k in range(N_STEPS)]

print("\n" + "="*70)
print("  EVALUATION — Step 1 forecast vs future_obs[0]  (held-out test)")
print("="*70)

# Step-1 comparison (hardest direct comparison)
obs_step1 = future_obs[0]
cases = {
    "Deterministic"            : det_fc[0],
    "Ensemble Mean (no MP)"    : mean_no_mp[0],
    "Ensemble Mean (+ LapMP)"  : mean_mp[0],
    "Ensemble Median (+ LapMP)": med_mp[0],
}
for name, fc in cases.items():
    print(f"\n  {name}")
    print(f"    MSE  : {mse(fc,obs_step1):.5f}")
    print(f"    RMSE : {rmse(fc,obs_step1):.5f}")
    print(f"    MAE  : {mae(fc,obs_step1):.5f}")
    print(f"    Bias : {bias(fc,obs_step1):.5f}")

# Note: ensemble mean may have *slightly* higher RMSE than deterministic at
# step 1 due to averaging over diverse members.  The key advantage is that
# ensemble RMSE grows SLOWER with lead time and spread-skill ≈ 1.
print(f"\n  RMSE across lead times:")
print(f"    Deterministic : {[f'{v:.4f}' for v in rmse_det]}")
print(f"    Ens (no MP)   : {[f'{v:.4f}' for v in rmse_nmp]}")
print(f"    Ens (+LapMP)  : {[f'{v:.4f}' for v in rmse_mp_l]}")
print(f"\n  Spread-Skill (no MP)  : {[f'{v:.3f}' for v in ss_no_mp]}")
print(f"  Spread-Skill (+LapMP) : {[f'{v:.3f}' for v in ss_mp]}")

# ============================================================
# 17. PROBABILITY EXCEEDANCE MAPS
# ============================================================
thresholds = [0.5, 2.0, 5.0]
prob_nmp   = {t: np.mean(ens_no_mp > np.log(t+0.1), axis=0) for t in thresholds}
prob_mp    = {t: np.mean(ens_mp    > np.log(t+0.1), axis=0) for t in thresholds}

# ============================================================
# PLOT A — Cascade Scale Decomposition
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Cascade Decomposition of Last Training Frame (8 Scales)", fontsize=13)
for j, (ax, sc) in enumerate(zip(axes.flat, obs_scales)):
    lim = max(abs(sc.min()), abs(sc.max()))
    im  = ax.imshow(sc, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"Scale {j+1}  (σ={2**j}px)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT B — Laplace Motion Perturbation Visualisation
# ============================================================
V_sample = perturb_motion_laplace(V, B_X, B_Y, smooth_sigma=4.0, decay=1.0)
dV       = V_sample - V

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Laplace Motion Perturbation Analysis", fontsize=13)
lim_v = max(abs(V[0]).max(), abs(V[1]).max())

titles = [("Original V_x",      V[0],        "RdBu_r"),
          (f"ΔV_x (b={B_X:.3f})",dV[0],       "coolwarm"),
          ("Perturbed V_x",      V_sample[0], "RdBu_r"),
          ("Original V_y",       V[1],        "RdBu_r"),
          (f"ΔV_y (b={B_Y:.3f})",dV[1],       "coolwarm"),
          ("Perturbed V_y",      V_sample[1], "RdBu_r")]

lim_d = max(abs(dV[0]).max(), abs(dV[1]).max()) + 1e-6
for ax, (title, data, cmap) in zip(axes.flat, titles):
    lim = lim_d if "Δ" in title else lim_v
    im  = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT B2 — Laplace distribution check
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Distribution of Laplace Motion Perturbation Values", fontsize=12)
for ax, comp, b, label in zip(axes,
                               [dV[0].flatten(), dV[1].flatten()],
                               [B_X, B_Y], ["ΔV_x", "ΔV_y"]):
    ax.hist(comp, bins=60, color='steelblue', alpha=0.7, density=True, label='Sampled')
    xs = np.linspace(comp.min(), comp.max(), 200)
    ax.plot(xs, (1/(2*b))*np.exp(-np.abs(xs)/b), 'r-', linewidth=2,
            label=f'Laplace(0, b={b:.3f})')
    ax.set_title(f"Perturbation {label}"); ax.set_xlabel("Δ pixels/step")
    ax.set_ylabel("Density"); ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT C — Observed (last train) / Future obs / Forecasts / Errors
# ============================================================
vmin = min(R_input_t.min(), future_obs[0].min())
vmax = max(R_input_t.max(), future_obs[0].max())

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("Step 1 Forecast vs Held-Out Future Observation", fontsize=13)

panels_top = [
    (R_input_t,      "Last train frame (t=0, model input)", "viridis", vmin, vmax),
    (future_obs[0],  "Ground truth  t+1  (held-out test)",  "viridis", vmin, vmax),
    (mean_mp[0],     "Ensemble Mean (+LapMP)",               "viridis", vmin, vmax),
]
lim_diff = max(abs(mean_no_mp[0]-future_obs[0]).max(),
               abs(mean_mp[0]-future_obs[0]).max())
panels_bot = [
    (mean_no_mp[0]-future_obs[0], "Ens Mean (no MP) − Future Obs",  "RdBu_r"),
    (mean_mp[0]-future_obs[0],    "Ens Mean (+MP) − Future Obs",    "RdBu_r"),
    (std_mp[0]-std_no_mp[0],      "Uncertainty Gain\n(+MP − no MP)","plasma"),
]
for ax, (data, title, cmap, v0, v1) in zip(axes[0], panels_top):
    im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
    ax.set_title(title, fontsize=10); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax, (data, title, cmap) in zip(axes[1], panels_bot):
    kw = dict(cmap=cmap, vmin=-lim_diff, vmax=lim_diff) if "Obs" in title else dict(cmap=cmap)
    im = ax.imshow(data, **kw); ax.set_title(title, fontsize=10); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT D — Uncertainty maps
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Ensemble Uncertainty (std) — Step 1", fontsize=13)
lim_std = max(std_no_mp[0].max(), std_mp[0].max())
for ax, data, title in zip(axes,
                            [std_no_mp[0], std_mp[0]],
                            ["Without Motion Perturbation",
                             "With Laplace Motion Perturbation"]):
    im = ax.imshow(data, cmap="plasma", vmin=0, vmax=lim_std)
    ax.set_title(title, fontsize=11); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT E — Probability of Exceedance
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Probability of Exceedance — Step 1  (top: no MP | bottom: +Laplace MP)",
    fontsize=12)
for col, t in enumerate(thresholds):
    for row, (prob, label) in enumerate([(prob_nmp, "no MP"), (prob_mp, "+LapMP")]):
        im = axes[row, col].imshow(prob[t][0], cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[row, col].set_title(f"P(R>{t} mm/h) [{label}]", fontsize=10)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT F — RMSE & Spread-Skill across lead times
# ============================================================
steps_label = [f"+{(k+1)*5}m" for k in range(N_STEPS)]
x = np.arange(N_STEPS); w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Metric Evolution across Lead Times  (vs held-out future obs)", fontsize=13)

axes[0].bar(x - w, rmse_det,  w, label="Deterministic",      color="coral")
axes[0].bar(x,     rmse_nmp,  w, label="Ens Mean (no MP)",    color="steelblue", alpha=0.8)
axes[0].bar(x + w, rmse_mp_l, w, label="Ens Mean (+LapMP)",   color="seagreen",  alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(steps_label)
axes[0].set_xlabel("Lead Time"); axes[0].set_ylabel("RMSE")
axes[0].set_title("RMSE: Deterministic vs Ensembles")
axes[0].legend(); axes[0].grid(axis='y', alpha=0.4)

axes[1].plot(range(1, N_STEPS+1), ss_no_mp, 'o--', color='steelblue',
             linewidth=2, markersize=8, label="No Motion Perturb")
axes[1].plot(range(1, N_STEPS+1), ss_mp,    's-',  color='seagreen',
             linewidth=2, markersize=8, label="+Laplace Motion Perturb")
axes[1].axhline(1.0, color='red', linestyle=':', linewidth=2, label='Ideal = 1.0')
axes[1].set_xlabel("Lead Time (steps)"); axes[1].set_ylabel("Spread / Skill")
axes[1].set_title("Spread-Skill Ratio vs Lead Time")
axes[1].legend(); axes[1].grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT G — Forecast Evolution
# ============================================================
fig, axes = plt.subplots(3, N_STEPS, figsize=(26, 10))
fig.suptitle(
    "Forecast Evolution — Det (top) | Ens Mean no-MP (mid) | Ens Mean +LapMP (bot)",
    fontsize=12)
rows = [(det_fc, "Det"), (mean_no_mp, "no MP"), (mean_mp, "+LapMP")]
for r, (data, lbl) in enumerate(rows):
    for k in range(N_STEPS):
        im = axes[r,k].imshow(data[k], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[r,k].set_title(f"{lbl} +{(k+1)*5}m", fontsize=8)
        axes[r,k].axis("off")
        plt.colorbar(im, ax=axes[r,k], fraction=0.046, pad=0.04)
plt.tight_layout(); plt.show()

# ============================================================
# PLOT H — Spaghetti Plot
# ============================================================
cy, cx = R_log.shape[1]//2, R_log.shape[2]//2

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle(f"Spaghetti Plot — Centre Pixel ({cx},{cy})", fontsize=13)
for ax, ens, mean_fc, title in zip(
        axes,
        [ens_no_mp, ens_mp],
        [mean_no_mp, mean_mp],
        ["Without Motion Perturbation", "With Laplace Motion Perturbation"]):
    for m in range(N_ENS):
        ax.plot(range(1, N_STEPS+1),
                [ens[m,k,cy,cx] for k in range(N_STEPS)],
                color='steelblue', alpha=0.2, linewidth=0.8)
    ax.plot(range(1, N_STEPS+1), [det_fc[k,cy,cx]  for k in range(N_STEPS)],
            'r-o', lw=2, label='Deterministic')
    ax.plot(range(1, N_STEPS+1), [mean_fc[k,cy,cx] for k in range(N_STEPS)],
            'k--o', lw=2, label='Ensemble Mean')
    # Plot future observations for the centre pixel  ← KEY FIX
    ax.plot(range(1, N_STEPS+1), [future_obs[k,cy,cx] for k in range(N_STEPS)],
            'g-s', lw=2, label='Future obs (ground truth)')
    ax.set_title(title); ax.set_xlabel("Lead Time (steps)")
    ax.set_ylabel("Log Rain Rate"); ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*72)
print("  MILESTONE 4 — FULL METRICS SUMMARY")
print("  Evaluated against held-out FUTURE frames (proper test set)")
print("="*72)
print(f"{'Metric':<22} {'Det':>10} {'Ens(noMP)':>11} {'Ens(+LapMP)':>13} {'Med(+MP)':>10}")
print("-"*72)
rows_m = [("MSE",  mse), ("RMSE", rmse), ("MAE", mae), ("Bias", bias)]
for mname, fn in rows_m:
    vd  = fn(det_fc[0],      obs_step1)
    vn  = fn(mean_no_mp[0],  obs_step1)
    vm  = fn(mean_mp[0],     obs_step1)
    vmd = fn(med_mp[0],      obs_step1)
    print(f"  {mname:<20} {vd:>10.5f} {vn:>11.5f} {vm:>13.5f} {vmd:>10.5f}")

ss1_nmp = ss(ens_no_mp[:,0], obs_step1)
ss1_mp  = ss(ens_mp[:,0],    obs_step1)
print(f"  {'Spread-Skill (step1)':<20} {'N/A':>10} {ss1_nmp:>11.4f} {ss1_mp:>13.4f} {'N/A':>10}")
print("="*72)
