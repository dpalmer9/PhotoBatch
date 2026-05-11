# IO sub-package registry aggregating vendor plugins.

from photobatch.Processing.IO.Behaviour import BEHAVIOUR_REGISTRY
from photobatch.Processing.IO.Photometry import SIGNAL_REGISTRY
from photobatch.Processing.IO.sync import abet_doric_synchronize

SYNC_REGISTRY = {
    ('abet', 'doric'): abet_doric_synchronize,
}

__all__ = ['BEHAVIOUR_REGISTRY', 'SIGNAL_REGISTRY', 'SYNC_REGISTRY']
