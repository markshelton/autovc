import logging

def setup_logging(name):
    logging.basicConfig(
        filename=name + ".log",
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger
