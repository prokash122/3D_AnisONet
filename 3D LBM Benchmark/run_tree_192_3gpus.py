#!/usr/bin/env python
"""Run the JAX_LBM D3Q19 solver over a whole *tree* of crop folders, on 3 GPUs at once.

For the layout crop_rotate_directions writes:

    rotated_crops_192/
        phi0.20_seed0/  dir00_....npy ... dir12_....npy  directions.json
        phi0.21_seed0/  ...
        manifest.json

Every .npy under every parent folder becomes one job. The parent process builds the full job
list, splits it round-robin across the GPUs, and launches one child per GPU pinned with
CUDA_VISIBLE_DEVICES. Each child writes its own CSV shard as it goes; the parent merges them
when all children exit.

    python run_tree_192_3gpus.py --root rotated_crops_192
    python run_tree_192_3gpus.py --root rotated_crops_192 --root rotated_crops_192_v2
    python run_tree_192_3gpus.py --root rotated_crops_192 --folders "phi0.2*" --dry-run
    python run_tree_192_3gpus.py --root rotated_crops_192 --fit          # + fit K per folder

Rows are keyed on "<root>/<folder>/<file>", not on the bare file name: every parent folder
holds a dir00_*.npy, so a basename key would collide and make folder 2..N look already-done.
Re-running resumes on that key (--no-resume to redo everything).

The solver is the one from JAX_LBM.ipynb (BGK collision, Zou-He pressure boundaries, solid
padding on transverse faces, Darcy post-processing) copied in, as in run_folder_192_3gpus.py.
Every boundary mask is a function argument, so one compiled kernel serves all 192^3 volumes
instead of silently reusing the first geometry's inlet/outlet masks.

Crops are (z, y, x) with flow along array axis 2 and 1 = solid, so --axis 2 (the default here)
and no --invert. That is the one default that differs from run_folder_192_3gpus.py.
"""
import argparse
import csv
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- LBM parameters
# (JAX_LBM.ipynb section 2)
TAU     = 1.0
NU      = 1.0 / 6.0
DELTA_P = 0.00005
RHO_IN  = 1.0
RHO_OUT = 1.0 - DELTA_P * 3.0
Q       = 19

MAX_STEPS   = 300000
PRINT_EVERY = 100
CONV        = 1e-4
CONV_WINDOW = 100
CONV_CONSEC = 2

CSV_FIELDS = ["key", "root", "folder", "file", "dir_index", "n_x", "n_y", "n_z",
              "axis", "nx", "ny", "nz", "porosity", "K_lattice", "K_mD",
              "u_darcy", "grad_P", "slice_std_rel", "steps", "converged",
              "K_rel_change", "energy_rel_change", "seconds", "gpu"]


# ------------------------------------------------------------------- job discovery
def root_label(root):
    return os.path.basename(os.path.abspath(root).rstrip(os.sep)) or "root"


def list_jobs(roots, pattern="*.npy", folder_globs=None):
    """Every .npy under every root, as dicts carrying its direction metadata.

    key = "<root name>/<path relative to the root>", so files with the same basename in
    different parent folders stay distinct.
    """
    jobs = []
    for root in roots:
        rl = root_label(root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
            rel_dir = "" if rel_dir == "." else rel_dir
            if folder_globs and rel_dir and not any(
                    fnmatch.fnmatch(rel_dir, g) for g in folder_globs):
                continue
            names = sorted(fnmatch.filter(filenames, pattern))
            if not names:
                continue
            meta = read_directions(dirpath)
            for name in names:
                m = meta.get(name, {})
                n = m.get("n", [None, None, None])
                jobs.append({
                    "key": f"{rl}/{rel_dir}/{name}" if rel_dir else f"{rl}/{name}",
                    "path": os.path.join(dirpath, name),
                    "root": rl, "folder": rel_dir, "file": name,
                    "dir_index": m.get("index", ""),
                    "n": n,
                })
    jobs.sort(key=lambda j: j["key"])
    return jobs


def read_directions(dirpath):
    """{file name -> entry} from a folder's directions.json, or {} if there is none."""
    p = os.path.join(dirpath, "directions.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {os.path.basename(c["file"]): c for c in data.get("crops", [])}


def shard_csv(csv_path, n, shard):
    return f"{csv_path[:-4]}.shard{shard}of{n}.csv"


def csv_glob(csv_path):
    return csv_path[:-4] + "*.csv"


def done_keys(csv_path):
    """{key -> row} across the merged CSV and every shard."""
    out = {}
    for p in sorted(glob.glob(csv_glob(csv_path))):
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("key"):
                        out[row["key"]] = row
        except OSError:
            continue
    return out


# ================================================================ worker (one GPU)
def run_worker(args):
    """Solve every job assigned to this shard. Runs with one GPU visible."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".80")

    import jax
    jax.config.update("jax_enable_x64", True)   # must precede any array creation
    import jax.numpy as jnp
    from jax import jit
    import numpy as np

    n_shards = int(os.environ.get("NUM_SHARDS", "1"))
    shard    = int(os.environ.get("SHARD_INDEX", "0"))
    gpu      = os.environ.get("GPU_ID", "?")
    print(f"[shard {shard}/{n_shards}] devices: {jax.devices()}", flush=True)

    # ------------------------------------------------------ D3Q19 lattice (section 3)
    C_LAT = np.array([
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
        [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]
    ], dtype=np.int32)
    W   = np.array([1./3.] + [1./18.]*6 + [1./36.]*12, dtype=np.float64)
    OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])

    cx4 = jnp.array(C_LAT[:, 0], dtype=jnp.float64).reshape(Q, 1, 1, 1)
    cy4 = jnp.array(C_LAT[:, 1], dtype=jnp.float64).reshape(Q, 1, 1, 1)
    cz4 = jnp.array(C_LAT[:, 2], dtype=jnp.float64).reshape(Q, 1, 1, 1)
    w4  = jnp.array(W, dtype=jnp.float64).reshape(Q, 1, 1, 1)
    cx2 = jnp.array(C_LAT[:, 0], dtype=jnp.float64).reshape(Q, 1, 1)
    cy2 = jnp.array(C_LAT[:, 1], dtype=jnp.float64).reshape(Q, 1, 1)
    cz2 = jnp.array(C_LAT[:, 2], dtype=jnp.float64).reshape(Q, 1, 1)
    w2  = jnp.array(W, dtype=jnp.float64).reshape(Q, 1, 1)
    opp = jnp.array(OPP)

    POS_X  = jnp.array([1, 7, 9, 11, 13])
    NEG_X  = jnp.array([2, 8, 10, 12, 14])
    ZERO_X = jnp.array([0, 3, 4, 5, 6, 15, 16, 17, 18])

    # ------------------------------------------------------ core functions (section 5)
    def eq3d(rho, ux, uy, uz):
        usq = ux*ux + uy*uy + uz*uz
        cu  = cx4*ux[None] + cy4*uy[None] + cz4*uz[None]
        return w4 * rho[None] * (1 + 3*cu + 4.5*cu*cu - 1.5*usq[None])

    def eq2d(rho, ux, uy, uz):
        usq = ux*ux + uy*uy + uz*uz
        cu  = cx2*ux[None] + cy2*uy[None] + cz2*uz[None]
        return w2 * rho[None] * (1 + 3*cu + 4.5*cu*cu - 1.5*usq[None])

    def macro(f):
        rho = jnp.sum(f, 0)
        return rho, jnp.sum(f*cx4, 0)/rho, jnp.sum(f*cy4, 0)/rho, jnp.sum(f*cz4, 0)/rho

    def stream(f):
        out = jnp.zeros_like(f)
        for q in range(Q):
            s = jnp.roll(f[q], C_LAT[q, 0], 0)
            s = jnp.roll(s,    C_LAT[q, 1], 1)
            s = jnp.roll(s,    C_LAT[q, 2], 2)
            out = out.at[q].set(s)
        return out

    # ------------------------------------------------- Zou-He pressure BC (section 6)
    def zou_he_inlet(f, inlet_mask):
        """x = 0: rebuild only the unknown (+x) populations at imposed RHO_IN."""
        s = f[:, 0, :, :]
        shp = s.shape[1:]
        rho_in = RHO_IN * jnp.ones(shp)
        ux_in = 1.0 - (jnp.sum(s[ZERO_X], 0) + 2*jnp.sum(s[NEG_X], 0)) / rho_in
        feq_face = eq2d(rho_in, ux_in, jnp.zeros(shp), jnp.zeros(shp))
        s_new = s
        for i, j in [(1, 2), (7, 8), (9, 10), (11, 12), (13, 14)]:
            s_new = s_new.at[i].set(
                jnp.where(inlet_mask[0], s[j] + feq_face[i] - feq_face[j], s[i]))
        return f.at[:, 0, :, :].set(s_new)

    def zou_he_outlet(f, outlet_mask):
        """x = NX-1: rebuild only the unknown (-x) populations at imposed RHO_OUT."""
        s = f[:, -1, :, :]
        shp = s.shape[1:]
        rho_out = RHO_OUT * jnp.ones(shp)
        ux_out = -1.0 + (jnp.sum(s[ZERO_X], 0) + 2*jnp.sum(s[POS_X], 0)) / rho_out
        feq_face = eq2d(rho_out, ux_out, jnp.zeros(shp), jnp.zeros(shp))
        s_new = s
        for i, j in [(2, 1), (8, 7), (10, 9), (12, 11), (14, 13)]:
            s_new = s_new.at[i].set(
                jnp.where(outlet_mask[0], s[j] + feq_face[i] - feq_face[j], s[i]))
        return f.at[:, -1, :, :].set(s_new)

    # ------------------------------------------------------- time step (section 7)
    @jit
    def one_step(f, is_fluid, is_bb, is_nd, inlet_mask, outlet_mask):
        _, ux, uy, uz = macro(f)
        rho = jnp.sum(f, 0)
        feq = eq3d(rho, ux, uy, uz)
        f_coll = jnp.where(is_bb, f[opp],
                 jnp.where(is_nd, f,
                           f - (f - feq) / TAU))
        f_str = stream(f_coll)
        f_str = zou_he_inlet(f_str, inlet_mask)
        return zou_he_outlet(f_str, outlet_mask)

    @jit
    def probe(f, is_fluid, y0, y1, z0, z1):
        """Superficial ux sum over the unpadded slab + mean fluid kinetic energy."""
        rho = jnp.sum(f, 0)
        ux  = jnp.sum(f*cx4, 0)/rho
        uy  = jnp.sum(f*cy4, 0)/rho
        uz  = jnp.sum(f*cz4, 0)/rho
        fluid = is_fluid[0]
        ny, nz = fluid.shape[1], fluid.shape[2]
        yy = jnp.arange(ny)[None, :, None]
        zz = jnp.arange(nz)[None, None, :]
        slab = (yy >= y0) & (yy < y1) & (zz >= z0) & (zz < z1)
        ux_sum = jnp.sum(jnp.where(fluid & slab, ux, 0.0))
        energy = 0.5 * rho * (ux**2 + uy**2 + uz**2)
        e_mean = jnp.sum(jnp.where(fluid, energy, 0.0)) / jnp.sum(fluid)
        return ux_sum, e_mean

    # ------------------------------------------------ geometry prep (section 4)
    def prepare_geometry(vol, flow_axis=2, invert=False):
        """Volume -> solver geometry with flow along axis 0, transverse faces walled.

        0 = fluid, 1 = bounce-back solid, 2 = no-dynamics, as in the notebook. stream()
        uses a periodic jnp.roll, so any transverse face carrying fluid gets a solid layer
        - otherwise opposite faces couple through the boundary.
        """
        vol = np.ascontiguousarray(vol)
        if invert:
            vol = (vol == 0).astype(np.uint8)
        if flow_axis == 1:
            vol = vol.transpose(1, 0, 2)
        elif flow_axis == 2:
            vol = vol.transpose(2, 1, 0)
        geo_raw = np.ascontiguousarray(vol).astype(np.uint8)
        phi = float(np.mean(geo_raw == 0))

        need_y = np.any(geo_raw[:, 0, :] == 0) or np.any(geo_raw[:, -1, :] == 0)
        need_z = np.any(geo_raw[:, :, 0] == 0) or np.any(geo_raw[:, :, -1] == 0)
        pad_y = (1, 1) if need_y else (0, 0)
        pad_z = (1, 1) if need_z else (0, 0)
        if any(pad_y) or any(pad_z):
            geo = np.pad(geo_raw, ((0, 0), pad_y, pad_z), mode="constant",
                         constant_values=1)
        else:
            geo = geo_raw
        return geo, pad_y, pad_z, phi

    def build_masks(geo):
        NX, NY, NZ = geo.shape
        is_fluid = jnp.array(geo == 0)[None]
        is_bb    = jnp.array(geo == 1)[None]
        is_nd    = jnp.array(geo == 2)[None]
        inm  = np.zeros((NY, NZ), bool)
        outm = np.zeros((NY, NZ), bool)
        inm[1:NY-1, 1:NZ-1]  = (geo[0,  1:NY-1, 1:NZ-1] == 0)
        outm[1:NY-1, 1:NZ-1] = (geo[-1, 1:NY-1, 1:NZ-1] == 0)
        return is_fluid, is_bb, is_nd, jnp.array(inm)[None], jnp.array(outm)[None]

    # -------------------------------------------- Darcy post-processing (section 10)
    def darcy_permeability(ux, geo, pad_y, pad_z):
        ux = np.asarray(ux)
        ys = slice(pad_y[0], geo.shape[1] - pad_y[1] if pad_y[1] else None)
        zs = slice(pad_z[0], geo.shape[2] - pad_z[1] if pad_z[1] else None)
        ux_slab, geo_slab = ux[:, ys, zs], geo[:, ys, zs]
        ux_zeroed = np.where(geo_slab == 0, ux_slab, 0.0)
        u_darcy = float(ux_zeroed.sum() / ux_slab.size)   # superficial: solid included
        nx_sim = ux_slab.shape[0]
        grad_p = DELTA_P / (nx_sim - 1)
        K = NU * u_darcy / grad_p
        per_slice = ux_zeroed.sum(axis=(1, 2)) / (ux_slab.shape[1] * ux_slab.shape[2])
        slice_std_rel = float(np.std(per_slice) / (abs(np.mean(per_slice)) + 1e-30))
        return dict(K=float(K), u_darcy=u_darcy, grad_P=float(grad_p),
                    slice_std_rel=slice_std_rel)

    # ------------------------------------------------ main loop (sections 8 & 9)
    def run_lbm(geo, pad_y, pad_z, label=""):
        NX, NY, NZ = geo.shape
        is_fluid, is_bb, is_nd, inlet_mask, outlet_mask = build_masks(geo)

        x = np.arange(NX, dtype=np.float64)
        rho0 = jnp.array(np.broadcast_to(
            (1.0 - DELTA_P*3/(NX-1)*x)[:, None, None], (NX, NY, NZ)).copy())
        zeros = jnp.zeros((NX, NY, NZ), dtype=jnp.float64)
        f = eq3d(rho0, zeros, zeros, zeros)

        y0, y1 = pad_y[0], NY - pad_y[1]
        z0, z1 = pad_z[0], NZ - pad_z[1]
        slab_size = NX * (y1 - y0) * (z1 - z0)
        grad_p = DELTA_P / (NX - 1)

        window_samples = max(2, args.conv_window // args.print_every)
        k_hist, e_hist = [], []
        prev_k = prev_e = None
        k_rel = e_rel = 1.0
        consec, converged, step = 0, False, 0
        t0 = time.time()

        for step in range(1, args.max_steps + 1):
            f = one_step(f, is_fluid, is_bb, is_nd, inlet_mask, outlet_mask)
            if step % args.print_every:
                continue

            ux_sum, e_now = probe(f, is_fluid, y0, y1, z0, z1)
            k_now = NU * (float(ux_sum) / slab_size) / grad_p
            e_now = float(e_now)
            k_hist.append(k_now)
            e_hist.append(e_now)
            k_win = float(np.mean(k_hist[-window_samples:]))
            e_win = float(np.mean(e_hist[-window_samples:]))
            if prev_k is not None and k_win != 0:
                k_rel = abs(k_win - prev_k) / abs(k_win)
            if prev_e is not None and e_win != 0:
                e_rel = abs(e_win - prev_e) / abs(e_win)

            # convergence is judged on K, the quantity actually being exported
            window_full = len(k_hist) >= window_samples
            consec = consec + 1 if (window_full and k_rel < args.conv) else 0

            if step % (args.print_every * 10) == 0 or consec >= CONV_CONSEC:
                print(f"    {label} step {step:6d}  K = {k_now:.6g}  "
                      f"dK/K = {k_rel:.2e}  dE/E = {e_rel:.2e}  "
                      f"{consec}/{CONV_CONSEC}", flush=True)

            if consec >= CONV_CONSEC and step > args.conv_window:
                converged = True
                break
            prev_k, prev_e = k_win, e_win

        _, ux, _, _ = macro(f)
        res = darcy_permeability(np.array(ux), geo, pad_y, pad_z)
        res.update(steps=step, converged=bool(converged),
                   K_rel_change=float(k_rel), energy_rel_change=float(e_rel),
                   seconds=float(time.time() - t0))
        return res

    # ------------------------------------------------------------------ job loop
    jobs = list_jobs(args.root, args.pattern, args.folders or None)
    mine = [j for i, j in enumerate(jobs) if i % n_shards == shard]
    have = {} if args.no_resume else done_keys(args.csv)

    out_path = shard_csv(args.csv, n_shards, shard)
    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()

    print(f"[shard {shard}] {len(mine)} of {len(jobs)} job(s) assigned", flush=True)
    t_start = time.time()
    for i, job in enumerate(mine, 1):
        key = job["key"]
        if key in have:
            print(f"[shard {shard}] ({i}/{len(mine)}) {key}: already done, skipped",
                  flush=True)
            continue
        try:
            vol = np.load(job["path"])
        except Exception as exc:
            print(f"[shard {shard}] ({i}/{len(mine)}) {key}: LOAD ERROR {exc}", flush=True)
            continue

        geo, pad_y, pad_z, phi = prepare_geometry(vol, args.axis, args.invert)
        print(f"[shard {shard}] ({i}/{len(mine)}) {key}  {geo.shape}  "
              f"porosity {phi:.4f}  pad y{pad_y} z{pad_z}", flush=True)
        try:
            res = run_lbm(geo, pad_y, pad_z, label=key)
        except Exception as exc:
            print(f"[shard {shard}] {key}: SOLVER ERROR {exc}", flush=True)
            continue

        k_md = ""
        if args.dx_um:
            k_m2 = res["K"] * (args.dx_um * 1e-6) ** 2
            k_md = f"{k_m2 / 9.869233e-16:.6g}"      # m^2 -> millidarcy
        n = job["n"]
        writer.writerow({
            "key": key, "root": job["root"], "folder": job["folder"], "file": job["file"],
            "dir_index": job["dir_index"],
            "n_x": "" if n[0] is None else f"{n[0]:.6f}",
            "n_y": "" if n[1] is None else f"{n[1]:.6f}",
            "n_z": "" if n[2] is None else f"{n[2]:.6f}",
            "axis": args.axis,
            "nx": geo.shape[0], "ny": geo.shape[1], "nz": geo.shape[2],
            "porosity": f"{phi:.6f}", "K_lattice": f"{res['K']:.6g}", "K_mD": k_md,
            "u_darcy": f"{res['u_darcy']:.6e}", "grad_P": f"{res['grad_P']:.6e}",
            "slice_std_rel": f"{res['slice_std_rel']:.4f}", "steps": res["steps"],
            "converged": int(res["converged"]),
            "K_rel_change": f"{res['K_rel_change']:.2e}",
            "energy_rel_change": f"{res['energy_rel_change']:.2e}",
            "seconds": f"{res['seconds']:.1f}", "gpu": gpu,
        })
        fh.flush()
        print(f"[shard {shard}] {key}: K = {res['K']:.6g}  "
              f"{'converged' if res['converged'] else 'NOT CONVERGED'} at "
              f"{res['steps']} steps in {res['seconds']:.0f}s", flush=True)

    fh.close()
    print(f"[shard {shard}] finished in {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


# ================================================================ parent (launcher)
def detect_gpus():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return [0]
    try:
        out = subprocess.run([exe, "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        ids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return ids or [0]
    except Exception:
        return [0]


def pump(stream, log_path, tag, lock):
    """Copy a child's stdout to its log file and to the console, tagged."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        for line in stream:
            fh.write(line)
            fh.flush()
            with lock:
                sys.stdout.write(f"[{tag}] {line.rstrip()}\n")
                sys.stdout.flush()


def merge_shards(csv_path, expected_keys):
    """Merge shard CSVs into one canonical file, keyed on <root>/<folder>/<file>."""
    shards = sorted(glob.glob(csv_path[:-4] + ".shard*.csv"))
    if not shards:
        print("no shard CSVs to merge")
        return
    rows = {}
    for p in ([csv_path] if os.path.exists(csv_path) else []) + shards:
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row.get("key"):
                        rows[row["key"]] = row
        except OSError:
            continue
    rank = {k: i for i, k in enumerate(expected_keys)}
    keep = [r for k, r in sorted(rows.items(), key=lambda kv: rank.get(kv[0], -1))
            if k in rank]
    if not keep:
        print("nothing to merge")
        return
    if os.path.exists(csv_path):
        shutil.copyfile(csv_path, csv_path + ".prev")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(keep)
    missing = [k for k in rank if k not in rows]
    print(f"merged {len(shards)} shard(s) -> {csv_path}: {len(keep)} row(s)")
    if missing:
        print(f"  {len(missing)} file(s) still have no row, e.g. {missing[0]}")
    else:
        print("  every file in the tree has exactly one row")


# ------------------------------------------------------------------- tensor fit
def fit_tensors(csv_path, out_path):
    """Least-squares K per folder from the merged CSV: k(n) = n^T K n, Mandel basis."""
    import numpy as np

    if not os.path.exists(csv_path):
        print(f"no CSV to fit: {csv_path}")
        return
    by_folder = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["n_x"] or not row["K_lattice"]:
                continue
            fkey = f"{row['root']}/{row['folder']}" if row["folder"] else row["root"]
            by_folder.setdefault(fkey, []).append(
                ([float(row["n_x"]), float(row["n_y"]), float(row["n_z"])],
                 float(row["K_lattice"]), float(row["porosity"])))

    s = np.sqrt(2.0)
    out = []
    for fkey, items in sorted(by_folder.items()):
        n = np.array([it[0] for it in items])
        k = np.array([it[1] for it in items])
        if len(n) < 6:
            print(f"  {fkey}: only {len(n)} direction(s), need >= 6 - skipped")
            continue
        A = np.column_stack([n[:, 0]**2, n[:, 1]**2, n[:, 2]**2,
                             s*n[:, 0]*n[:, 1], s*n[:, 1]*n[:, 2], s*n[:, 0]*n[:, 2]])
        sol, *_ = np.linalg.lstsq(A, k, rcond=None)
        K = np.array([[sol[0], sol[3]/s, sol[5]/s],
                      [sol[3]/s, sol[1], sol[4]/s],
                      [sol[5]/s, sol[4]/s, sol[2]]])
        rel = float(np.linalg.norm(A @ sol - k)/max(1e-300, np.linalg.norm(k)))
        ev = np.linalg.eigvalsh(K)
        out.append({
            "folder": fkey, "n_dirs": len(n),
            "porosity": f"{np.mean([it[2] for it in items]):.6f}",
            "Kxx": f"{K[0,0]:.6g}", "Kyy": f"{K[1,1]:.6g}", "Kzz": f"{K[2,2]:.6g}",
            "Kxy": f"{K[0,1]:.6g}", "Kyz": f"{K[1,2]:.6g}", "Kxz": f"{K[0,2]:.6g}",
            "k1": f"{ev[0]:.6g}", "k2": f"{ev[1]:.6g}", "k3": f"{ev[2]:.6g}",
            "anisotropy": f"{ev[2]/ev[0]:.4f}" if ev[0] > 0 else "",
            "cond_A": f"{np.linalg.cond(A):.3f}", "rel_residual": f"{rel:.3e}",
        })
        print(f"  {fkey}: {len(n)} dirs  K_diag = ({K[0,0]:.4g}, {K[1,1]:.4g}, "
              f"{K[2,2]:.4g})  k3/k1 = {ev[2]/ev[0] if ev[0] > 0 else float('nan'):.3f}"
              f"  resid {rel:.1e}")
    if not out:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {len(out)} tensor(s) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="crop root holding one sub-folder per parent geometry; "
                         "repeat for several roots (default rotated_crops_192)")
    ap.add_argument("--folders", action="append", default=None,
                    help="glob on the sub-folder name, e.g. 'phi0.2*'; repeatable")
    ap.add_argument("--pattern", default="*.npy")
    ap.add_argument("--csv", default="permeability_tree.csv")
    ap.add_argument("--gpus", default="0,1,2",
                    help="comma-separated GPU ids, one child each (default 0,1,2; "
                         "'auto' for every card nvidia-smi reports)")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                    help="array axis the flow runs along (default 2: crops are (z,y,x) "
                         "with flow along x)")
    ap.add_argument("--invert", action="store_true",
                    help="volumes store 1 = pore, 0 = solid")
    ap.add_argument("--dx-um", type=float, default=10.0,
                    help="voxel size in microns, for the K_mD column (0 to skip)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--print-every", type=int, default=PRINT_EVERY)
    ap.add_argument("--conv", type=float, default=CONV)
    ap.add_argument("--conv-window", type=int, default=CONV_WINDOW)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--mem-fraction", default=".80",
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION per child (one card each)")
    ap.add_argument("--fit", action="store_true",
                    help="after merging, fit K per folder into --fit-csv")
    ap.add_argument("--fit-csv", default="permeability_tensors.csv")
    ap.add_argument("--fit-only", action="store_true",
                    help="skip the solver, just fit K from an existing --csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    # internal: set by the parent when it launches a child
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not args.root:
        args.root = ["rotated_crops_192"]

    if args.worker:
        return run_worker(args)

    if args.fit_only:
        fit_tensors(args.csv, args.fit_csv)
        return 0

    for r in args.root:
        if not os.path.isdir(r):
            sys.exit(f"root not found: {os.path.abspath(r)}")
    gpus = detect_gpus() if args.gpus == "auto" else \
        [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    n = len(gpus)
    jobs = list_jobs(args.root, args.pattern, args.folders or None)
    if not jobs:
        sys.exit(f"no files matching {args.pattern} under {[os.path.abspath(r) for r in args.root]}")
    keys = [j["key"] for j in jobs]
    os.makedirs(args.logdir, exist_ok=True)

    folders = {}
    for j in jobs:
        folders.setdefault(f"{j['root']}/{j['folder']}", 0)
        folders[f"{j['root']}/{j['folder']}"] += 1
    counts = sorted(set(folders.values()))
    split = [sum(1 for i in range(len(jobs)) if i % n == s) for s in range(n)]

    print(f"roots    : {', '.join(os.path.abspath(r) for r in args.root)}")
    print(f"folders  : {len(folders)}   crops per folder: "
          f"{counts[0] if len(counts) == 1 else f'{counts[0]}-{counts[-1]} (uneven)'}")
    print(f"jobs     : {len(jobs)}  (flow along axis {args.axis})")
    print(f"gpus     : {gpus}")
    print(f"split    : {split} across {n} shard(s), each file run exactly once")
    if len(counts) > 1:
        odd = [f for f, c in sorted(folders.items()) if c != counts[0]]
        print(f"  note: {len(odd)} folder(s) have a different crop count, e.g. "
              f"{odd[0]} ({folders[odd[0]]})")
    if not args.no_resume:
        have = done_keys(args.csv)
        n_done = sum(1 for k in keys if k in have)
        if n_done:
            print(f"resume   : {n_done} already have a CSV row and will be skipped "
                  f"(--no-resume to redo)")
    print(f"logs     : {os.path.abspath(args.logdir)}\n")

    base = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
            "--pattern", args.pattern, "--csv", args.csv,
            "--axis", str(args.axis), "--dx-um", str(args.dx_um),
            "--max-steps", str(args.max_steps), "--print-every", str(args.print_every),
            "--conv", str(args.conv), "--conv-window", str(args.conv_window)]
    for r in args.root:
        base += ["--root", r]
    for g in (args.folders or []):
        base += ["--folders", g]
    if args.invert:
        base.append("--invert")
    if args.no_resume:
        base.append("--no-resume")

    procs, threads = [], []
    lock = threading.Lock()
    t0 = time.time()
    for shard, gpu in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # each child sees exactly one card
        env["NUM_SHARDS"] = str(n)
        env["SHARD_INDEX"] = str(shard)
        env["GPU_ID"] = str(gpu)
        env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.mem_fraction

        log_path = os.path.join(args.logdir, f"gpu{gpu}.tree.log")
        print(f"  gpu {gpu} -> shard {shard}/{n}  ->  {shard_csv(args.csv, n, shard)}, "
              f"stdout -> {log_path}")
        if args.dry_run:
            print(f"      CUDA_VISIBLE_DEVICES={gpu} NUM_SHARDS={n} SHARD_INDEX={shard}")
            print(f"      {' '.join(base)}")
            continue
        p = subprocess.Popen(base, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append((gpu, shard, p))
        th = threading.Thread(target=pump,
                              args=(p.stdout, log_path, f"gpu{gpu}", lock), daemon=True)
        th.start()
        threads.append(th)

    if args.dry_run:
        print(f"\nfirst 3 jobs: {keys[:3]}")
        print(f"last  job   : {keys[-1]}")
        return 0

    print(f"\n{len(procs)} process(es) running in parallel. Ctrl-C to stop all.")
    print("(first output appears once a child has compiled its kernel)\n")
    try:
        for _, _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\ninterrupted - terminating children")
        for _, _, p in procs:
            p.terminate()
        for _, _, p in procs:
            p.wait()
    for th in threads:
        th.join(timeout=5)

    print(f"\nall done in {(time.time()-t0)/60:.1f} min")
    bad = 0
    for gpu, shard, p in procs:
        bad += p.returncode != 0
        print(f"  gpu {gpu} shard {shard}: "
              f"{'ok' if p.returncode == 0 else f'FAILED (exit {p.returncode})'}   "
              f"log: {args.logdir}/gpu{gpu}.tree.log")
    print()
    if not args.no_merge:
        merge_shards(args.csv, keys)
    if args.fit:
        print()
        fit_tensors(args.csv, args.fit_csv)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
