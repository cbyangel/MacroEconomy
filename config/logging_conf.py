import logging

def setup_logger(name: str = None, level=logging.INFO):
    """
    공통 logging 설정을 반환한다.
    모든 모듈(loader / indicators / label / normalize)에서 import 해서 사용.

    example:
        from config.logging_conf import setup_logger
        logger = setup_logger(__name__)
        logger.info("hi")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # 출력 format
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        # console output
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger