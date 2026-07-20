from __future__ import annotations

import argparse
import logging

from voicecommander.app import VoiceCommander, configure_logging
from voicecommander.settings import CONFIG_PATH, Settings, load_settings, local_runtime_available, show_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceCommander")
    parser.add_argument("--settings", action="store_true", help="edit settings and exit")
    parser.add_argument("--captions", action="store_true", help="live-caption system audio in a window")
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    open_settings = args.settings
    try:
        config = load_settings()
    except (OSError, ValueError):
        logger.exception("Configuration is invalid; opening settings with defaults")
        config = Settings()
        open_settings = True

    if args.captions:
        from voicecommander.captions import run_captions

        return run_captions(config)

    needs_setup = not CONFIG_PATH.exists() or (
        config.asr_provider == "local"
        and not local_runtime_available(config.local_asr_model)
    )
    if open_settings or needs_setup:
        updated = show_settings(config)
        if updated is None:
            logger.info("Settings closed without saving")
            return 0 if open_settings else 1
        config = updated
        if open_settings:
            return 0

    return VoiceCommander(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
