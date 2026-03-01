# CARD/utils/log.py
import logging

logger = logging.getLogger("CARD")
logger.setLevel(logging.INFO)

# Add console log handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)
