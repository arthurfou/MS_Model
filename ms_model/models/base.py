MODEL_REGISTRY = {}


def register_model(name: str):
    """Décorateur : enregistre une classe de modèle sous `name` pour build_model."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, **kwargs):
    """Instancie un modèle enregistré par nom (cf. config["model"]["name"])."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu: '{name}'. Disponibles: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
