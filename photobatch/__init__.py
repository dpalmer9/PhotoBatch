"""photobatch — Fiber photometry batch analysis pipeline."""

import logging

# Library best practice: attach NullHandler so callers that don't configure
# logging don't see "No handlers could be found for logger 'photobatch'" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def configure_logging(level=logging.INFO, log_file="photobatch.log"):
    """Configure the photobatch logger with a StreamHandler and a FileHandler.

    Call this once at application startup (e.g. in the GUI launch script or
    before calling process_files) so that all pipeline log records are routed
    to the console and to *log_file*.

    Parameters
    ----------
    level : int
        Logging level (default logging.INFO).
    log_file : str
        Path for the rotating file log (default 'photobatch.log' in the CWD).
    """
    pkg_logger = logging.getLogger("photobatch")
    pkg_logger.setLevel(level)

    if pkg_logger.handlers:
        # Already configured – avoid adding duplicate handlers
        return

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    pkg_logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        pkg_logger.addHandler(fh)
    except OSError as exc:
        pkg_logger.warning("Could not open log file %r: %s", log_file, exc)
