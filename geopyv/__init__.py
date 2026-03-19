import importlib
import logging

# Core modules (must succeed).
import geopyv.log
import geopyv.check
import geopyv.image
import geopyv._image_extensions
import geopyv.io
import geopyv.object
import geopyv.subset
import geopyv._subset_extensions
import geopyv.templates
import geopyv.validation

# Optional modules – imported lazily so that missing dependencies
# (e.g. geomat, PySide6) do not prevent the core library from loading.
_optional_modules = [
    "geopyv.bayes",
    "geopyv.calibration",
    "geopyv.chain",
    "geopyv.field",
    "geopyv.geometry",
    "geopyv.gui",
    "geopyv.mesh",
    "geopyv.particle",
    "geopyv.plots",
    "geopyv.sequence",
    "geopyv.speckle",
]

for _mod_name in _optional_modules:
    try:
        importlib.import_module(_mod_name)
    except Exception:
        pass

# Initialise log at default settings.
level = logging.INFO
geopyv.log.initialise(level)
log = logging.getLogger(__name__)
log.debug("Initialised geopyv log.")
