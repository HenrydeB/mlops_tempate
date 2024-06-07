import logging.config
from pathlib import Path

from rich.logging import RichHandler

logging.config.fileConfig(Path(__file__).parent / "log_config_file.config")
logger = logging.getLogger(__name__)
logger.root.handlers[0] = RichHandler(markup=True)


logger.debug("Debug")
logger.info("Information")
logger.warning("Warning")
logger.error("Error")
logger.critical("Critical Error")
