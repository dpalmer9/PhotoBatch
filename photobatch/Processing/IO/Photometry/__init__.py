# Photometry loader sub-package registry.

from .doric import load_doric_data

SIGNAL_REGISTRY = {
    'doric': {
        'load': load_doric_data,
    }
}
