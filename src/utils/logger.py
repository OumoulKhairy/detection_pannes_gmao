"""Messages horodatés, identiques dans les notebooks et les scripts."""
import logging
import sys


def get_logger(nom="detection_pannes_gmao"):
    logger = logging.getLogger(nom)
    if not logger.handlers:                              # évite les doublons si on relance la cellule
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()
