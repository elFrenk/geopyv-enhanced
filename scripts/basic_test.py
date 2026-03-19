"""
Basic geopyv test — subset-level DIC on two Jet images.

Usage:
    python scripts/basic_test.py
    python scripts/basic_test.py --no-gui          # headless (no matplotlib windows)
    python scripts/basic_test.py --save-plots       # save plots to PNG instead of showing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Basic geopyv subset test")
parser.add_argument("--no-gui", action="store_true", help="Disable interactive plots")
parser.add_argument("--save-plots", action="store_true", help="Save convergence plot to PNG")
args, _ = parser.parse_known_args()

# Force non-interactive backend before any matplotlib import.
if args.no_gui or args.save_plots:
    os.environ["MPLBACKEND"] = "Agg"

# WSL safety settings.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import geopyv as gp


def main() -> None:
    images_dir = PROJECT_ROOT / "images" / "Jet"
    ref_path = images_dir / "Jet_0001A.jpg"
    tar_path = images_dir / "Jet_0001B.jpg"

    if not ref_path.exists() or not tar_path.exists():
        sys.exit(
            f"Test images not found in {images_dir}.\n"
            "Copy them from the original geopyv repo:\n"
            "  cp ../geopyv/images/Jet/Jet_0001A.jpg ../geopyv/images/Jet/Jet_0001B.jpg images/Jet/"
        )

    # ── 1. Load images ──────────────────────────────────────────────
    print("Loading images…")
    ref = gp.image.Image(str(ref_path))
    tar = gp.image.Image(str(tar_path))
    print(f"  Reference: {ref_path.name}  shape={ref.image_gs.shape}")
    print(f"  Target:    {tar_path.name}  shape={tar.image_gs.shape}")

    # ── 2. Create template and subset ────────────────────────────────
    template = gp.templates.Circle(50)
    coord = np.array([500.0, 500.0])

    print(f"\nCreating subset at coord={coord} with radius=50…")
    subset = gp.subset.Subset(
        f_img=ref,
        g_img=tar,
        f_coord=coord,
        template=template,
    )

    # ── 3. Solve ────────────────────────────────────────────────────
    print("Solving (ICGN)…")
    subset.solve(method="ICGN", max_iterations=50)

    res = subset.data["results"]
    print(f"\n{'='*40}")
    print(f"  Solved:      {subset.data['solved']}")
    print(f"  u (px):      {res['u']:.4f}")
    print(f"  v (px):      {res['v']:.4f}")
    print(f"  Iterations:  {res['iterations']}")
    print(f"  Norm:        {res['norm']:.2e}")
    print(f"  C_ZNCC:      {res['C_ZNCC']:.4f}")
    print(f"  C_ZNSSD:     {res['C_ZNSSD']:.4f}")
    print(f"{'='*40}")

    # ── 4. Save / reload ────────────────────────────────────────────
    out_file = "test_subset"
    gp.io.save(object=subset, filename=out_file)
    loaded = gp.io.load(filename=out_file)
    assert loaded.data["solved"], "Loaded subset not marked as solved!"
    print(f"\nSave/load round-trip OK  →  {out_file}.pyv")

    # ── 5. Optional convergence plot ─────────────────────────────────
    if args.save_plots:
        import importlib
        import matplotlib.pyplot as plt

        # Ensure plots module is loaded (may have been skipped at init).
        if not hasattr(gp, "plots"):
            importlib.import_module("geopyv.plots")
        subset.convergence()
        fig = plt.gcf()
        plot_path = PROJECT_ROOT / "convergence.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Convergence plot saved to {plot_path}")
    elif not args.no_gui:
        try:
            import importlib

            if not hasattr(gp, "plots"):
                importlib.import_module("geopyv.plots")
            subset.convergence()
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass  # If no display available, just skip.

    print("\nAll done — basic test passed!")


if __name__ == "__main__":
    main()
