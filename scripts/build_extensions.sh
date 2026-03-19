#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# build_extensions.sh — Compile the pybind11 C++ extensions for the
#                       active Python interpreter.
#
# Usage:
#   ./scripts/build_extensions.sh                   # auto-detect Eigen
#   EIGEN_DIR=/path/to/eigen ./scripts/build_extensions.sh
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/geopyv"

# ── Eigen include path ──────────────────────────────────────────
if [[ -z "${EIGEN_DIR:-}" ]]; then
    # Try sibling geopyv repo first, then system.
    if [[ -d "$PROJECT_ROOT/../geopyv/external/eigen" ]]; then
        EIGEN_DIR="$PROJECT_ROOT/../geopyv/external/eigen"
    elif pkg-config --cflags eigen3 &>/dev/null; then
        EIGEN_DIR=""  # system headers via pkg-config
    else
        echo "ERROR: Eigen not found."
        echo "  Set EIGEN_DIR=/path/to/eigen or install libeigen3-dev."
        exit 1
    fi
fi

EIGEN_FLAGS=""
if [[ -n "${EIGEN_DIR}" ]]; then
    EIGEN_FLAGS="-I${EIGEN_DIR}"
else
    EIGEN_FLAGS="$(pkg-config --cflags eigen3)"
fi

# ── pybind11 & Python flags ────────────────────────────────────
PYBIND_FLAGS=$(python3 -m pybind11 --includes)
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Project root : $PROJECT_ROOT"
echo "Eigen flags  : $EIGEN_FLAGS"
echo "pybind flags : $PYBIND_FLAGS"
echo "Extension    : $EXT_SUFFIX"
echo ""

# ── Compile ─────────────────────────────────────────────────────
for module in _image_extensions _subset_extensions; do
    src="$SRC_DIR/${module/_extensions/}.cpp"     # _image.cpp / _subset.cpp
    out="$SRC_DIR/${module}${EXT_SUFFIX}"

    echo "Building ${module}…"
    g++ -O2 -shared -std=c++14 -fPIC -fopenmp -DEIGEN_DONT_PARALLELIZE \
        $PYBIND_FLAGS $EIGEN_FLAGS \
        "$src" -o "$out"
    echo "  → $out"
done

echo ""
echo "Done. Verify with:  python -c \"from geopyv import _image_extensions, _subset_extensions; print('OK')\""
