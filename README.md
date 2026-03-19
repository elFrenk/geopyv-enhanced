# geopyv-enhanced

Fork potenziata di [geopyv](https://github.com/sas229/geopyv) — Digital Image Correlation (DIC) / PIV in Python.

---

## Setup da zero

### Prerequisiti di sistema (Ubuntu/WSL)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev g++ libomp-dev
```

### 1. Clona il repo e il repo originale (serve per Eigen headers)

```bash
cd ~/projects
git clone https://github.com/elFrenk/geopyv-enhanced.git
git clone https://github.com/sas229/geopyv.git        # serve per external/eigen
cd geopyv-enhanced
```

> **Nota:** se hai già Eigen installato a sistema (`libeigen3-dev`), puoi saltare il clone di geopyv e usare `EIGEN_DIR=/usr/include/eigen3`.

### 2. Crea il virtual environment e installa le dipendenze

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Compila le estensioni C++ (pybind11 + Eigen)

```bash
./scripts/build_extensions.sh
```

Se Eigen è in un path custom:
```bash
EIGEN_DIR=/path/to/eigen ./scripts/build_extensions.sh
```

Verifica che funzioni:
```bash
python -c "from geopyv import _image_extensions, _subset_extensions; print('OK')"
```

### 4. Copia le immagini di test

```bash
mkdir -p images/Jet
cp ../geopyv/images/Jet/Jet_0001A.jpg ../geopyv/images/Jet/Jet_0001B.jpg images/Jet/
```

### 5. Lancia il test base

```bash
python scripts/basic_test.py --no-gui
```

Con salvataggio plot:
```bash
python scripts/basic_test.py --save-plots
```

---

## Struttura del progetto

```
geopyv/              # Libreria core (DIC/PIV)
  _image.cpp         # Estensione C++ per pre-calcolo B-spline
  _subset.cpp        # Estensione C++ per correlazione subset
  image.py           # Classe Image
  subset.py          # Classe Subset (DIC)
  mesh.py            # Classe Mesh (full-field)
  sequence.py        # Classe Sequence (multi-frame)
  templates.py       # Circle/Square templates
  io.py              # Save/load (.pyv files)
  plots.py           # Visualizzazione risultati
scripts/
  basic_test.py      # Test base funzionante
  build_extensions.sh # Script di compilazione C++
  run_pipeline.py    # Pipeline completa (WIP)
images/              # Immagini di test
configs/             # Configurazioni YAML
```

---

## Note importanti

- Le estensioni `.so` **devono** essere compilate per la stessa versione di Python del venv. Se cambi versione Python, ricompila con `./scripts/build_extensions.sh`.
- Il modulo `particle` richiede `geomat` che non si installa su Python ≥3.12. Non blocca il funzionamento core.
- Su WSL: il backend matplotlib è forzato a `Agg` (no GUI). Usa `--save-plots` per salvare i grafici.
