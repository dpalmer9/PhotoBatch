# IO sub-package for PhotoBatch
# Contains vendor-specific loaders (Photometry/, Behaviour/),
# synchronization (sync.py) and output writing (output.py).

from photobatch.Processing.IO.Behaviour.abet import (
    load_abet_data,
    abet_extract_information,
)
from photobatch.Processing.IO.Photometry.doric import load_doric_data
from photobatch.Processing.IO.sync import abet_doric_synchronize

# Plugin registries for vendor-dispatched loading
BEHAVIOUR_REGISTRY = {
    'abet': {
        'load': load_abet_data,
        'extract_info': abet_extract_information,
    },
}

SIGNAL_REGISTRY = {
    'doric': {
        'load': load_doric_data,
    },
}

SYNC_REGISTRY = {
    ('abet', 'doric'): abet_doric_synchronize,
}
