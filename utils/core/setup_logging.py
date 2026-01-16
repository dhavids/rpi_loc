import logging
import os
import sys
from datetime import datetime
from pathlib import Path

MAX_EXPERIMENT_LOGS = 3

# Separator formatting constants
SEP_WIDTH = 50
SEP_CHAR = "-"


class ShortFormatter(logging.Formatter):
    """
    Custom formatter for short logs.
    Format: time [TYPE] [name]: msg
    - Uses context_name if set via get_named_logger(), shows both context and module source
    - Strips __ from module names
    - Uses only last element of dotted names
    """
    def format(self, record):
        # Get short module name: last element of dotted name, strip underscores
        module_name = record.name.split('.')[-1].strip('_')
        
        # Use context_name if available, combine with module source
        context_name = getattr(record, 'context_name', None)
        if context_name is not None and context_name != module_name:
            name = f"{context_name}:{module_name}"
        else:
            name = module_name
        
        # Format timestamp with milliseconds
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S,%f")[:12]
        
        # Format level (shorter)
        level = record.levelname
        
        # Build the message
        msg = record.getMessage()
        return f"{time_str} [{level}] [{name}]: {msg}"


class LongFormatter(logging.Formatter):
    """
    Custom formatter for full logs output.
    Format: time [TYPE] [process] [name]: msg
    - Uses context_name if set via get_named_logger(), shows both context and module source
    """
    def format(self, record):
        # Format timestamp with milliseconds
        time_str = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:23]
        
        # Format level
        level = record.levelname.ljust(8)
        
        # Get short module name for display
        module_name = record.name.split('.')[-1].strip('_')
        
        # Use context_name if available, combine with module source
        context_name = getattr(record, 'context_name', None)
        if context_name is not None and context_name != module_name:
            name = f"{context_name}:{module_name}"
        else:
            name = module_name
        
        # Build the message
        msg = record.getMessage()
        return f"{time_str} [{level}] [{record.process}] [{name}]: {msg}"


class ContextNameFilter(logging.Filter):
    """
    Filter that adds a context_name attribute to log records.
    Used by get_named_logger() to provide custom logger names.
    """
    def __init__(self, context_name: str):
        super().__init__()
        self.context_name = context_name
    
    def filter(self, record):
        record.context_name = self.context_name
        return True


def get_named_logger(context_name: str, module_name: str = None) -> logging.Logger:
    """
    Get a logger with a custom context name displayed in log output.
    
    This allows logs to show a meaningful name like [tb3_mpe] instead of
    the module path [translator.src.base_trainer].
    
    Args:
        context_name: The name to display in log output (e.g., "tb3_mpe")
        module_name: Optional module name for the underlying logger. 
                     If None, uses context_name as the logger name.
    
    Returns:
        A logger that displays context_name in formatted output.
    
    Example:
        logger = get_named_logger("tb3_mpe", __name__)
        logger.info("Training started")  # Output: 12:00:00 [INFO] [tb3_mpe]: Training started
    """
    logger_name = module_name or context_name
    logger = logging.getLogger(logger_name)
    
    # Add filter if not already present
    for f in logger.filters:
        if isinstance(f, ContextNameFilter) and f.context_name == context_name:
            return logger  # Already configured
    
    logger.addFilter(ContextNameFilter(context_name))
    return logger


def _section_log(self, level, msg, args, exc_info=None, extra=None, 
                 stack_info=False, stacklevel=1, **kwargs):
    """
    Custom _log method that supports sep/bottom kwargs for section formatting.
    
    Usage:
        logger.info("Episode 0 started", sep=True)           # Separator line above message
        logger.info("Episode complete", bottom=True)         # Short separator below message
        logger.info("Major section", sep=True, bottom=True)  # Both above and below
        logger.info("Regular message")                       # Normal log line
        logger.info("Conditional msg", log=some_flag)        # Only logs if log=True or level > INFO
    """
    # Extract our custom kwargs
    sep = kwargs.pop('sep', False)
    bottom = kwargs.pop('bottom', False)
    n_dashes = kwargs.pop('n_dashes', SEP_WIDTH)
    should_log = kwargs.pop('log', True)

    # Legacy suport for 'verbose' kwarg
    if should_log:
        should_log = kwargs.pop('verbose', True)

    # Legacy support for 'decorator' kwarg
    decorator = kwargs.pop('decorator', None)
    if isinstance(decorator, str):
        decorator = decorator.lower()
        if decorator == "info":
            level = logging.INFO
        elif decorator in ("warn", "warning"):
            level = logging.WARNING
        elif decorator == "error":
            level = logging.ERROR
    
    # Skip logging only for DEBUG and INFO levels if log=False
    # WARNING, ERROR, CRITICAL always log regardless of flag
    if not should_log and level <= logging.INFO:
        return
    
    # Use the original _log method stored on the class
    original_log = logging.Logger._original_log
    
    # Top separator
    if sep:
        original_log(
            self, level, SEP_CHAR * n_dashes, (), exc_info, extra, 
            stack_info, stacklevel)
    
    # Main message
    original_log(
        self, level, msg, args, exc_info, extra, stack_info, 
        stacklevel)
    
    # Bottom separator (shorter)
    if bottom:
        original_log(
            self, level, SEP_CHAR * max(1, n_dashes // 5), (), 
            exc_info, extra, stack_info, stacklevel)


# Monkey-patch the Logger class to support sep/bottom/log kwargs - should work for all loggers
if not hasattr(logging.Logger, '_original_log'):
    logging.Logger._original_log = logging.Logger._log
    logging.Logger._log = _section_log


def setup_logging(
    experiment_name: str = "rpi_loc",
    log_dir: str = "logs",
    log_file: str = None,
    level: str = "info",
    log_to_file: bool = True,
    log_to_console: bool = True,
    verbose: bool = False,
):
    """
    Setup logging for rpi_loc.
    
    Args:
        experiment_name: Name for the experiment/log grouping
        log_dir: Directory for log files (relative or absolute)
        log_file: Explicit log file path (overrides log_dir)
        level: Log level (debug, info, warning, error, critical)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        verbose: Print log file location
    
    Returns:
        Path to the log file if log_to_file=True, else None
    """
    # If we are logging to file, we try to resolve the log file first
    if log_to_file:
        if log_file is not None:
            if not os.path.isabs(log_file):
                print("Log file path must be absolute if provided directly."
                      f"Recieved: {log_file}")
                return None
            
            log_file = Path(log_file)
            log_dir = log_file.parent

            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Failed to create log directory: {log_dir}\n{e}")
                return None
        else:
            # If log directory is not a full path, 
            # place logs in <rpi_loc>/files/logs/<experiment_name>
            if not os.path.isabs(log_dir):
                # This file is in <rpi_loc>/utils/core/setup_logging.py
                # Resolve to: rpi_loc/files/logs/<experiment_name>
                rpi_loc_root = Path(__file__).resolve().parent.parent.parent
                log_dir = rpi_loc_root / "files" / "logs" / experiment_name
            else:
                log_dir = Path(log_dir)
            
            log_dir.mkdir(parents=True, exist_ok=True)

            # Only add experiment_name if not already in folder name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if experiment_name in log_dir.name:
                log_file = log_dir / f"{timestamp}.log"
                prune_pattern = "*.log"  # Prune all logs in this directory
            else:
                log_file = log_dir / f"{experiment_name}_{timestamp}.log"
                prune_pattern = f"{experiment_name}_*.log"

            _prune_old_logs(log_dir, prune_pattern)

    # Create formatters - we use short for both now
    file_formatter = ShortFormatter()
    console_formatter = ShortFormatter()

    root_logger = logging.getLogger()

    if isinstance(level, str):
        level = level.lower()
        level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(level, logging.INFO)

    root_logger.setLevel(level)

    # Prevent duplicate handlers (important for notebooks / restarts)
    if root_logger.handlers:
        return log_file

    # File handler (single experiment file)
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

    if log_to_file:
        root_logger.addHandler(file_handler)
        if verbose:
            logging.getLogger(__name__).info(
                "Logs for experiment: %s", experiment_name)
            print(f"{'-'*50}\nLog file: {log_file}")
    
    if log_to_console:
        root_logger.addHandler(console_handler)

    return log_file


def _prune_old_logs(log_dir: Path, pattern: str):
    logs = sorted(
        log_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for old_log in logs[MAX_EXPERIMENT_LOGS:]:
        try:
            old_log.unlink()
        except Exception:
            pass
