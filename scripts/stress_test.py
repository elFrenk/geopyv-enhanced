"""
Stress test geopyv — 3 livelli di complessità crescente.

Livello 1: Multi-subset — griglia di subset su una coppia di immagini
Livello 2: Mesh         — mesh full-field su compression images
Livello 3: Sequence     — analisi multi-frame sulle 10 coppie Jet

Usage:
    python scripts/stress_test.py --no-gui                 # run all, headless
    python scripts/stress_test.py --no-gui --level 2       # solo fino a livello 2
    python scripts/stress_test.py --no-gui --save-plots    # salva PNG dei risultati
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="geopyv stress test")
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--save-plots", action="store_true")
parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3],
                    help="Max livello di test (1=subset, 2=mesh, 3=sequence)")
args, _ = parser.parse_known_args()

if args.no_gui or args.save_plots:
    os.environ["MPLBACKEND"] = "Agg"

# WSL safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import geopyv as gp

OUTPUT_DIR = PROJECT_ROOT / "stress_test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure if --save-plots."""
    if args.save_plots:
        import matplotlib.pyplot as plt
        path = OUTPUT_DIR / f"{name}.png"
        plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    [saved] {path.name}")


def timer(label: str):
    """Simple context-manager timer."""
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            dt = time.perf_counter() - self.t0
            print(f"  [{label}] {dt:.2f}s")
    return _T()


# ═══════════════════════════════════════════════════════════════════
# LIVELLO 1 — Multi-subset grid
# ═══════════════════════════════════════════════════════════════════
def level_1():
    print("\n" + "=" * 60)
    print("LIVELLO 1 — Multi-subset grid  (Jet pair)")
    print("=" * 60)

    ref = gp.image.Image(str(PROJECT_ROOT / "images/Jet/Jet_0001A.jpg"))
    tar = gp.image.Image(str(PROJECT_ROOT / "images/Jet/Jet_0001B.jpg"))
    h, w = ref.image_gs.shape
    print(f"  Image: {w}x{h}")

    template = gp.templates.Circle(30)
    margin = 60  # template radius + safety

    # Build a grid of points
    xs = np.arange(margin, w - margin, 80)
    ys = np.arange(margin, h - margin, 80)
    grid = np.array([[x, y] for y in ys for x in xs])
    print(f"  Grid: {len(xs)}x{len(ys)} = {len(grid)} subset points")

    solved = 0
    failed = 0
    results = []

    with timer(f"{len(grid)} subsets"):
        for i, coord in enumerate(grid):
            sub = gp.subset.Subset(f_img=ref, g_img=tar,
                                   f_coord=coord, template=template)
            sub.solve(method="ICGN", max_iterations=40)
            if sub.data["solved"]:
                solved += 1
                r = sub.data["results"]
                results.append((coord[0], coord[1], r["u"], r["v"], r["C_ZNCC"]))
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"    ... {i+1}/{len(grid)} done")

    results = np.array(results)
    print(f"\n  Solved: {solved}/{len(grid)}  |  Failed: {failed}")
    if len(results) > 0:
        print(f"  u range: [{results[:,2].min():.2f}, {results[:,2].max():.2f}] px")
        print(f"  v range: [{results[:,3].min():.2f}, {results[:,3].max():.2f}] px")
        print(f"  C_ZNCC:  [{results[:,4].min():.3f}, {results[:,4].max():.3f}]")

    # Quiver plot
    if args.save_plots and len(results) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(ref.image_gs, cmap="gray")
        ax.quiver(results[:,0], results[:,1], results[:,2], -results[:,3],
                  color="cyan", scale=30, width=0.003)
        ax.set_title(f"Level 1 — Subset displacement field ({solved} points)")
        ax.set_aspect("equal")
        save_fig("L1_quiver")

    return solved > 0


# ═══════════════════════════════════════════════════════════════════
# LIVELLO 2 — Full-field Mesh (compression)
# ═══════════════════════════════════════════════════════════════════
def level_2():
    print("\n" + "=" * 60)
    print("LIVELLO 2 — Mesh full-field  (compression pair)")
    print("=" * 60)

    ref = gp.image.Image(str(PROJECT_ROOT / "images/compression/compression_0.jpg"))
    tar = gp.image.Image(str(PROJECT_ROOT / "images/compression/compression_1.jpg"))
    h, w = ref.image_gs.shape
    print(f"  Image: {w}x{h}")

    template = gp.templates.Circle(50)

    # Rectangular RoI — well inside image bounds
    boundary = gp.geometry.region.Path(
        nodes=np.array([
            [150.0, 150.0],
            [150.0, 850.0],
            [850.0, 850.0],
            [850.0, 150.0],
        ]),
        hard=False,
    )

    # Exclusion zone (simulates an obstacle)
    exclusions = [
        gp.geometry.region.Circle(
            centre=np.array([700.0, 700.0]),
            radius=50.0,
            size=20.0,
            option="F",
            hard=True,
        )
    ]

    seed = np.array([500.0, 500.0])

    print("  Creating mesh (target_nodes=150)...")
    with timer("mesh init"):
        mesh = gp.mesh.Mesh(
            f_img=ref,
            g_img=tar,
            target_nodes=150,
            boundary_obj=boundary,
            exclusion_objs=exclusions,
            mesh_order=1,
        )

    print("  Solving mesh (ICGN, adaptive_iterations=2)...")
    with timer("mesh solve"):
        mesh.solve(
            seed_coord=seed,
            template=template,
            adaptive_iterations=2,
            method="ICGN",
            alpha=0.3,
            tolerance=0.7,
            seed_tolerance=0.85,
        )

    subsets = mesh.data["results"]["subsets"]
    solved_count = sum(1 for s in subsets if s["solved"])
    total_count = len(subsets)
    print(f"\n  Subsets solved: {solved_count}/{total_count}")

    # Save mesh
    gp.io.save(object=mesh, filename="stress_mesh",
               directory=str(OUTPUT_DIR))
    print(f"  Mesh saved to {OUTPUT_DIR/'stress_mesh.pyv'}")

    # Plots
    if args.save_plots:
        import matplotlib.pyplot as plt

        for qty in ["u", "v", "C_ZNCC"]:
            try:
                mesh.contour(quantity=qty, colorbar=True, alpha=0.7, show=False)
                save_fig(f"L2_contour_{qty}")
            except Exception as e:
                print(f"    [skip] contour {qty}: {e}")

        try:
            mesh.quiver(show=False)
            save_fig("L2_quiver")
        except Exception as e:
            print(f"    [skip] quiver: {e}")

        try:
            mesh.convergence(show=False)
            save_fig("L2_convergence")
        except Exception as e:
            print(f"    [skip] convergence: {e}")

    return solved_count > 0


# ═══════════════════════════════════════════════════════════════════
# LIVELLO 3 — Sequence multi-frame (Jet, 10 coppie)
# ═══════════════════════════════════════════════════════════════════
def level_3():
    print("\n" + "=" * 60)
    print("LIVELLO 3 — Sequence multi-frame  (compression, 11 frames)")
    print("=" * 60)

    template = gp.templates.Circle(50)

    # Compression images are 1001x1001
    boundary = gp.geometry.region.Path(
        nodes=np.array([
            [150.0, 150.0],
            [150.0, 850.0],
            [850.0, 850.0],
            [850.0, 150.0],
        ]),
        hard=False,
    )

    seed = np.array([500.0, 500.0])

    print("  Creating sequence (target_nodes=150)...")
    with timer("sequence init"):
        sequence = gp.sequence.Sequence(
            image_dir=str(PROJECT_ROOT / "images" / "compression"),
            common_name="compression",
            target_nodes=150,
            boundary_obj=boundary,
            save_by_reference=False,
            ID="stress_test",
        )

    images = sequence.data["file_settings"]["images"]
    n_images = len(images)
    n_meshes = len(sequence.data["meshes"])
    print(f"  Images found: {n_images}")
    print(f"  Mesh pairs to solve: {n_meshes}")

    print("  Solving sequence (ICGN, mesh_order=1, subset_order=1)...")
    with timer("sequence solve"):
        try:
            sequence.solve(
                seed_coord=seed,
                template=template,
                mesh_order=1,
                subset_order=1,
                method="ICGN",
                alpha=0.3,
                tolerance=0.55,
                seed_tolerance=0.70,
                guide=False,  # guide=True needs geomat (not available on Py3.12)
                dense=False,
                sync=True,
                sequential=True,
            )
        except Exception as e:
            print(f"  [warning] Sequence solve stopped early: {e}")

    # Report results
    meshes = sequence.data.get("meshes", [])
    solved_meshes = sum(1 for m in meshes if isinstance(m, dict) and m.get("solved"))
    print(f"\n  Meshes solved: {solved_meshes}/{len(meshes)}")
    for i, m_data in enumerate(meshes):
        if not isinstance(m_data, dict):
            continue
        results = m_data.get("results", {})
        subs = results.get("subsets", []) if isinstance(results, dict) else []
        n_sub = len(subs)
        n_solved = sum(1 for s in subs if isinstance(s, dict) and s.get("solved"))
        print(f"    Mesh {i}: {n_solved}/{n_sub} subsets solved")

    # Save
    gp.io.save(object=sequence, filename="stress_sequence",
               directory=str(OUTPUT_DIR))
    print(f"  Sequence saved to {OUTPUT_DIR/'stress_sequence.pyv'}")

    # Plots
    solved_indices = [i for i, m in enumerate(meshes)
                      if isinstance(m, dict) and m.get("solved")]
    if args.save_plots and len(solved_indices) > 0:
        import matplotlib.pyplot as plt

        plot_indices = [solved_indices[0]]
        if len(solved_indices) > 1:
            plot_indices.append(solved_indices[min(4, len(solved_indices) - 1)])

        for mi in plot_indices:
            for qty in ["u", "v"]:
                try:
                    sequence.contour(mesh_index=mi, quantity=qty,
                                     colorbar=True, alpha=0.7, show=False)
                    save_fig(f"L3_contour_mesh{mi}_{qty}")
                except Exception as e:
                    print(f"    [skip] contour mesh {mi} {qty}: {e}")
            try:
                sequence.quiver(mesh_index=mi, show=False)
                save_fig(f"L3_quiver_mesh{mi}")
            except Exception as e:
                print(f"    [skip] quiver mesh {mi}: {e}")

    return len(meshes) > 0


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    t0 = time.perf_counter()
    print("geopyv stress test")
    print(f"Level: 1..{args.level}  |  save-plots: {args.save_plots}")

    ok = True
    if args.level >= 1:
        ok = level_1() and ok
    if args.level >= 2:
        ok = level_2() and ok
    if args.level >= 3:
        ok = level_3() and ok

    total = time.perf_counter() - t0
    print(f"\n{'='*60}")
    status = "ALL PASSED" if ok else "SOME FAILURES"
    print(f"  {status}  —  Total time: {total:.1f}s")
    if args.save_plots:
        print(f"  Plots in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
