import logging
from typing import ClassVar

from sty import ef, fg, rs  # type: ignore


class DuplicateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # add other fields if you need more granular comparison, depends on your app
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


class CustomFormatter(logging.Formatter):
    CUSTOM_FORMAT = "%(asctime)s.%(msecs)d %(levelname)s:%(name)s: %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: fg.grey + CUSTOM_FORMAT + rs.all,
        logging.INFO: fg.white + CUSTOM_FORMAT + rs.all,
        logging.WARNING: fg.red + CUSTOM_FORMAT + rs.all,
        logging.ERROR: ef.bold + fg.li_red + CUSTOM_FORMAT + rs.all,
        logging.CRITICAL: ef.bold + fg.magenta + CUSTOM_FORMAT + rs.all,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def getLogger(name: str, log_level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(name)
    log.addFilter(DuplicateFilter())
    syslog = logging.StreamHandler()
    syslog.setFormatter(CustomFormatter())
    log.addHandler(syslog)
    log.setLevel(log_level)
    return log
