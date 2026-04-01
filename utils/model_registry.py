"""
Model registry — manages model creation via the registry pattern.
"""
from typing import Dict, Callable, Any


class ModelRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable):
            self._registry[name] = func
            return func
        return decorator

    def create_model(self, name: str, p: Dict[str, Any]):
        if name not in self._registry:
            raise ValueError(f'Unknown model: {name}')
        return self._registry[name](p)


model_registry = ModelRegistry()


@model_registry.register('FourierPET')
def create_fourierpet_model(p: Dict[str, Any]):
    from models.FourierPET import create_FourierPET
    kwargs = p['backbone_kwargs']
    return create_FourierPET(
        admm_iter=kwargs['admm_iter'],
        inner_iter=kwargs['inner_iter'],
        pbeam_config=p['projection_kwargs'],
        mode1=kwargs['mode1'],
        mode2=kwargs['mode2'],
        hidden_dim=kwargs['hidden_dim'],
        ssd_state_dim=kwargs['ssd_state_dim'],
        wave=kwargs['wave'],
        J=kwargs['J'],
        spatial_dilations=kwargs['spatial_dilations'],
        ffn_dim=kwargs['ffn_dim'],
    )
