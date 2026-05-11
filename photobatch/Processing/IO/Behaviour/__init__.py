# Behaviour loader sub-package registry.

from .abet import (
    load_abet_data,
    abet_extract_information,
    abet_trial_definition,
    abet_search_event,
)

BEHAVIOUR_REGISTRY = {
    'abet': {
        'load': load_abet_data,
        'extract_info': abet_extract_information,
        'trial_definition': abet_trial_definition,
        'search_event': abet_search_event,
    }
}
