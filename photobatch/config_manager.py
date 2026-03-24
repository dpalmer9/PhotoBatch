"""photobatch.config_manager
============================
Replaces the previous monolithic ``Config.ini`` / ``configparser`` setup with
a distributed, typed JSON configuration system.

Design
------
* Each processing module owns its own JSON config file, co-located beside its
  ``.py`` source (e.g. ``Processing/Signal/filter.json`` lives next to
  ``Processing/Signal/filter.py``).
* A lightweight top-level ``photobatch/config.json`` holds global settings
  (file paths, output flags, concurrency, vendor selection).
* ``ConfigManager`` loads all files at startup, merges them into a single
  ``{section: {key: typed_value}}`` dictionary, and exposes a dict-like API
  that is backward-compatible with the ``config[section][key]`` access pattern
  the rest of the codebase already uses.
* ``ConfigManager.save()`` writes each section back to its *source* JSON file
  so edits made via the GUI are persisted in the right place.
* Vendor discovery (``discover_vendors``) scans ``IO/Behaviour/`` and
  ``IO/Photometry/`` for JSON files that carry a ``vendor_name`` field, making
  it straightforward to add new data sources by dropping a JSON file alongside
  a new IO module.

Usage
-----
::

    from photobatch.config_manager import ConfigManager

    config = ConfigManager(base_dir)   # base_dir = photobatch/ package root
    value  = config['Signal_Filter']['filter_type']   # 'lowpass'
    config['Signal_Filter']['filter_type'] = 'smoothing'
    config.save()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Section → JSON file path (relative to the photobatch/ package root)
# ---------------------------------------------------------------------------
# Keys are the section names exposed via __getitem__.
# Values are paths relative to the photobatch/ package root directory.
# Vendor sections (ABET, Doric, …) are loaded dynamically based on the active
# vendor selection in config.json and therefore are NOT listed here — they are
# added at runtime by _load_active_vendors().

SECTION_FILE_MAP: dict[str, str] = {
    # ---- global / main config ----
    "Filepath":    "config.json",
    "Output":      "config.json",
    "Concurrency": "config.json",
    "Vendors":     "config.json",

    # ---- process / event ----
    "Event_Window":  "Processing/Process/event.json",
    "Normalization": "Processing/Process/event.json",

    # ---- signal ----
    "Signal_Filter":     "Processing/Signal/filter.json",
    "Signal_Utilities":  "Processing/Signal/utilities.json",
    "Signal_Fitting":    "Processing/Signal/fitting.json",
}

# Sections that should be skipped when auto-generating UI groups in the
# Options tab (they are handled elsewhere or are internal).
HIDDEN_SECTIONS = {"Vendors"}

# Vendor category → subdirectory (relative to photobatch/)
VENDOR_DIRS: dict[str, str] = {
    "Behaviour": "Processing/IO/Behaviour",
    "Photometry": "Processing/IO/Photometry",
}


class _SectionProxy(dict):
    """A plain dict that also remembers which JSON file it came from."""

    def __init__(self, data: dict, source_file: Path):
        super().__init__(data)
        self._source_file = source_file


class ConfigManager:
    """Distributed JSON configuration manager.

    Parameters
    ----------
    base_dir : str or Path
        Absolute path to the ``photobatch/`` package root directory.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir).resolve()
        # Ordered mapping: section_name -> _SectionProxy
        self._data: dict[str, _SectionProxy] = {}
        # Track which file each section came from: section_name -> Path
        self._section_source: dict[str, Path] = {}
        # Cache of already-parsed JSON files: abs_path -> raw dict
        self._file_cache: dict[Path, dict] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _abs(self, rel: str) -> Path:
        return self._base / rel

    def _parse_json(self, abs_path: Path) -> dict:
        """Load and cache a JSON file, stripping ``_description`` keys."""
        if abs_path in self._file_cache:
            return self._file_cache[abs_path]
        with open(abs_path, encoding="utf-8") as fh:
            raw: dict = json.load(fh)
        # Remove description-only keys at top level (they start with _)
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        self._file_cache[abs_path] = raw
        return raw

    def _register_section(self, section: str, data: dict, source_file: Path) -> None:
        """Add or overwrite a section in the internal store."""
        # Strip inner _description keys
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        self._data[section] = _SectionProxy(clean, source_file)
        self._section_source[section] = source_file

    def _load_file_sections(self, abs_path: Path) -> None:
        """Parse a JSON file and register every top-level dict as a section."""
        raw = self._parse_json(abs_path)
        # Top-level keys that map to dicts are sections.
        # Top-level keys that map to scalar/non-dict values are metadata
        # (vendor_name, vendor_category) and are ignored as sections.
        for key, value in raw.items():
            if isinstance(value, dict):
                self._register_section(key, value, abs_path)

    def _load_all(self) -> None:
        """Load the main config.json and then all sub-configs."""
        main_file = self._abs("config.json")
        self._load_file_sections(main_file)

        loaded_files: set[Path] = {main_file}
        for rel_path in SECTION_FILE_MAP.values():
            abs_path = self._abs(rel_path)
            if abs_path in loaded_files:
                continue
            loaded_files.add(abs_path)
            try:
                self._load_file_sections(abs_path)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"Warning: Could not load config '{abs_path}': {exc}")

        self._load_active_vendors()

    def _load_active_vendors(self) -> None:
        """Load the active behaviour and signal vendor config sections."""
        vendors_section = self._data.get("Vendors", {})
        behaviour_vendor = vendors_section.get("behaviour_vendor", "")
        signal_vendor = vendors_section.get("signal_vendor", "")

        for category, subdir in VENDOR_DIRS.items():
            active_name = behaviour_vendor if category == "Behaviour" else signal_vendor
            if not active_name:
                continue
            vendors = self.discover_vendors(category)
            for vinfo in vendors:
                if vinfo["vendor_name"] == active_name:
                    self._load_file_sections(Path(vinfo["file"]))
                    break

    # ------------------------------------------------------------------
    # Public dict-like API
    # ------------------------------------------------------------------

    def sections(self) -> list[str]:
        """Return an ordered list of all section names."""
        return list(self._data.keys())

    def __getitem__(self, section: str) -> _SectionProxy:
        return self._data[section]

    def __setitem__(self, section: str, value: dict) -> None:
        source = self._section_source.get(section, self._abs("config.json"))
        self._register_section(section, value, source)

    def __contains__(self, section: object) -> bool:
        return section in self._data

    def get(self, section: str, key: str | None = None,
            default: Any = None) -> Any:
        """Two-argument form ``get(section, key, default)`` returns a value.
        One-argument form ``get(section)`` returns the section dict or None.
        """
        if key is None:
            return self._data.get(section)
        sec = self._data.get(section)
        if sec is None:
            return default
        return sec.get(key, default)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write each section back to its source JSON file."""
        # Group sections by source file
        by_file: dict[Path, dict[str, dict]] = {}
        for section, proxy in self._data.items():
            src = self._section_source.get(section, self._abs("config.json"))
            by_file.setdefault(src, {})[section] = dict(proxy)

        for abs_path, sections_to_write in by_file.items():
            # Preserve top-level non-section metadata (vendor_name, etc.)
            try:
                with open(abs_path, encoding="utf-8") as fh:
                    existing: dict = json.load(fh)
            except (FileNotFoundError, json.JSONDecodeError):
                existing = {}

            # Keep metadata keys (non-dict values and _description)
            merged = {k: v for k, v in existing.items()
                      if not isinstance(v, dict) or k.startswith("_")}
            merged.update(sections_to_write)

            with open(abs_path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, indent=4)

        # Invalidate cache so a subsequent load picks up the written values
        self._file_cache.clear()

    # ------------------------------------------------------------------
    # Vendor discovery / switching
    # ------------------------------------------------------------------

    def discover_vendors(self, category: str) -> list[dict]:
        """Scan the vendor subdirectory for registered vendor JSON files.

        Parameters
        ----------
        category : str
            ``'Behaviour'`` or ``'Photometry'``.

        Returns
        -------
        list of dicts with keys ``vendor_name``, ``vendor_category``, ``file``.
        """
        subdir = VENDOR_DIRS.get(category)
        if subdir is None:
            return []

        vendor_dir = self._base / subdir
        found: list[dict] = []
        if not vendor_dir.is_dir():
            return found

        for json_file in sorted(vendor_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                if "vendor_name" in data and "vendor_category" in data:
                    found.append({
                        "vendor_name": data["vendor_name"],
                        "vendor_category": data["vendor_category"],
                        "file": str(json_file),
                    })
            except (json.JSONDecodeError, OSError):
                pass
        return found

    def load_vendor(self, category: str, vendor_name: str) -> bool:
        """Swap in the configuration sections from a different vendor.

        Removes the sections that came from the *previously active* vendor's
        JSON file (identified by vendor_category) and loads the new vendor's
        sections.

        Parameters
        ----------
        category : str
            ``'Behaviour'`` or ``'Photometry'``.
        vendor_name : str
            Must match the ``vendor_name`` field in the vendor's JSON file.

        Returns
        -------
        bool  True if the vendor was found and loaded; False otherwise.
        """
        vendors = self.discover_vendors(category)
        target = next((v for v in vendors if v["vendor_name"] == vendor_name), None)
        if target is None:
            return False

        new_file = Path(target["file"])

        # Drop sections that came exclusively from the old vendor file for this
        # category.  We identify them by having a matching vendor_category in
        # the raw JSON.
        sections_to_remove = []
        for section, src in self._section_source.items():
            # Only remove if the source file is a vendor file in the same category
            # *and* it is not the new file (avoid removing what we're about to add).
            if str(src) == str(new_file):
                continue
            try:
                with open(src, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("vendor_category") == category:
                    sections_to_remove.append(section)
            except (OSError, json.JSONDecodeError):
                pass

        for s in sections_to_remove:
            self._data.pop(s, None)
            self._section_source.pop(s, None)

        # Load the new vendor file
        self._file_cache.pop(new_file, None)  # force re-read
        self._load_file_sections(new_file)

        # Update Vendors section
        if "Vendors" in self._data:
            key = "behaviour_vendor" if category == "Behaviour" else "signal_vendor"
            self._data["Vendors"][key] = vendor_name

        return True

    # ------------------------------------------------------------------
    # Serialization helpers (for snapshot diff in GUI)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict]:
        """Return a plain nested dict of all sections and their values."""
        return {section: dict(proxy) for section, proxy in self._data.items()}

    def to_json_string(self) -> str:
        """Serialize the entire merged config to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
