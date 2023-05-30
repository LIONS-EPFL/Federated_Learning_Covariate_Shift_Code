import logging


def getLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not len(logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    
    return logger
