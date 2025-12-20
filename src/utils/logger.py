import logging

def logTool(name: str):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s : %(asctime)s] : %(message)s",
    )
    return logging.getLogger(name)