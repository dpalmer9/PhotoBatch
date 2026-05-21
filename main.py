import argparse
import json
import logging
import sys
from pathlib import Path

from photobatch import configure_logging


LOGGER = logging.getLogger("photobatch.cli")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "PhotoBatch: Batch processing pipeline for fiber photometry and "
            "behavioural data."
        )
    )
    parser.add_argument(
        "--file-sheet", "-f",
        help="Path to the CSV file defining file pairs.",
    )
    parser.add_argument(
        "--event-sheet", "-e",
        help="Path to the CSV file defining behavioral events to analyze.",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to the JSON configuration file or the photobatch config directory.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory where output files will be saved (overrides config).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes for parallel batch execution.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run analysis pipeline in CLI-only headless mode.",
    )
    return parser


def _should_run_headless(args):
    return bool(args.headless or args.file_sheet or args.event_sheet or args.config)


def _resolve_config_base(config_arg):
    default_base = Path(__file__).resolve().parent / "photobatch"

    if config_arg:
        config_path = Path(config_arg).expanduser().resolve()
    else:
        config_path = default_base / "config.json"

    if config_path.is_dir():
        return config_path, None
    if config_path.is_file():
        if config_path.name == "config.json":
            return config_path.parent, config_path
        return default_base, config_path
    raise FileNotFoundError(f"Config path not found: {config_path}")


def _load_config(config_arg):
    from photobatch.config_manager import ConfigManager

    config_base, config_file = _resolve_config_base(config_arg)
    config = ConfigManager(config_base)

    if config_file and config_file.name != "config.json":
        with open(config_file, encoding="utf-8") as fh:
            raw_config = json.load(fh)
        for section, section_value in raw_config.items():
            if isinstance(section_value, dict) and not section.startswith("_"):
                config[section] = section_value

    return config_base, config


def _select_output_options(config):
    output_section = config.get("Output", default={}) or {}
    output_options = [
        index + 1
        for index, enabled in enumerate(output_section.values())
        if bool(enabled)
    ]
    if output_options:
        return output_options
    return list(range(1, len(output_section) + 1)) if output_section else []


def _run_headless(args, parser):
    if not (args.file_sheet and args.event_sheet):
        parser.error("Headless mode requires --file-sheet and --event-sheet.")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1.")

    from photobatch.Processing.data_processor import process_files

    config_base, config = _load_config(args.config)

    if args.output_dir:
        config["Filepath"]["output_path"] = str(Path(args.output_dir).expanduser().resolve())

    num_workers = args.workers
    if num_workers is None:
        try:
            num_workers = int(config.get("Concurrency", "num_workers", 1))
        except (TypeError, ValueError):
            num_workers = 1

    output_options = _select_output_options(config)
    LOGGER.info("Starting headless batch processing")
    LOGGER.info("Config base: %s", config_base)
    LOGGER.info("Selected outputs: %s", output_options)
    LOGGER.info("Workers: %s", num_workers)

    results_path = process_files(
        file_sheet_path=str(Path(args.file_sheet).expanduser().resolve()),
        event_sheet_path=str(Path(args.event_sheet).expanduser().resolve()),
        output_options=output_options,
        config=config,
        num_workers=num_workers,
    )
    print(f"Batch processing completed successfully. Results database: {results_path}")
    return 0


def _run_gui():
    try:
        from photobatch.GUI.launch import launch_photobatch
    except ImportError:
        print(
            'Error: GUI dependencies are not installed. Install PhotoBatch with the "gui" extra to launch the interface.',
            file=sys.stderr,
        )
        return 1

    launch_photobatch()
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(level=logging.INFO)

    if _should_run_headless(args):
        return _run_headless(args, parser)
    return _run_gui()


if __name__ == "__main__":
    raise SystemExit(main())