from typing import Optional
import logging

import torch.distributed as dist


def init_root_logger(output_path: str = "."):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)5s] {%(filename)s:%(lineno)03d} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            # logging.FileHandler(f"{output_path}/log.txt"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def update_logger_formatter_for_rank(logger, disable_print=False):
    if dist.is_initialized():
        rank = dist.get_rank()
        new_format = f"%(asctime)s [%(levelname)5s - Rank {rank}] {{%(filename)s:%(lineno)04d}} %(message)s"
        if disable_print:
            logger.setLevel(logging.ERROR)
            # logger.debug = lambda *args, **kwargs: None
            # logger.info = lambda *args, **kwargs: None
    else:
        new_format = (
            "%(asctime)s [%(levelname)5s] {%(filename)s:%(lineno)03d} %(message)s"
        )

    formatter = logging.Formatter(new_format, datefmt="%Y-%m-%d %H:%M:%S")

    if logger.handlers:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    else:
        from logging import _handlerList

        for handler_weak_ref in _handlerList:
            handler = handler_weak_ref()
            if handler is not None:
                handler.setFormatter(formatter)


init_root_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)


def set_logger_level(level: int, logger: Optional[logging.Logger] = None):
    logger = logger or get_logger()
    logger.setLevel(level)
