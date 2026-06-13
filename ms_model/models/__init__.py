from .base import build_model  # noqa: F401

# Importer chaque modèle pour qu'il s'enregistre dans MODEL_REGISTRY (cf. base.py)
from . import convlstm  # noqa: F401,E402
