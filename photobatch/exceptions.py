"""photobatch.exceptions
Custom exception hierarchy for the PhotoBatch pipeline.
"""


class PhotobatchError(Exception):
    """Base class for all PhotoBatch pipeline errors."""


class SynchronizationError(PhotobatchError):
    """Raised when TTL-based time synchronization between behaviour and photometry fails."""


class MissingColumnError(PhotobatchError):
    """Raised when a required column is absent from a loaded DataFrame."""


class UnsupportedFileFormatError(PhotobatchError):
    """Raised when a file format or vendor software version is not supported."""
