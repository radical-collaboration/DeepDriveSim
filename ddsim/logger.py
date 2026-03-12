import sys
from datetime import datetime
from enum import Enum


class Colors:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    def __init__(self, name="DDSimManager", use_colors=True, output_stream=None):
        self.name = name
        self.use_colors = use_colors
        self.output_stream = output_stream or sys.stdout

        self.level_colors = {
            LogLevel.DEBUG: Colors.BRIGHT_BLACK,
            LogLevel.INFO: Colors.BRIGHT_CYAN,
            LogLevel.WARNING: Colors.BRIGHT_YELLOW,
            LogLevel.ERROR: Colors.BRIGHT_RED,
            LogLevel.CRITICAL: Colors.RED + Colors.BOLD,
        }

        self.component_colors = {
            "task": Colors.BRIGHT_GREEN,
            "cancelation": Colors.BRIGHT_RED,
            "workflow": Colors.MAGENTA,
            "simulation": Colors.BLUE,
            "training": Colors.BRIGHT_YELLOW,
            "manager": Colors.GREEN,
            "evaluate": Colors.BRIGHT_MAGENTA,
            "finalization": Colors.BRIGHT_BLUE,
        }

    def _colorize(self, text, color):
        return f"{color}{text}{Colors.RESET}" if self.use_colors else text

    def _format_message(self, level, component, message, task_name=None):
        timestamp = self._colorize(
            datetime.now().strftime("%H:%M:%S.%f")[:-3], Colors.DIM
        )
        colored_level = self._colorize(
            f"[{self.name}-{level.value}]", self.level_colors.get(level, Colors.WHITE)
        )

        # Handle task-specific components
        if component.lower().startswith("task-"):
            component_color = Colors.BRIGHT_GREEN
        else:
            component_color = self.component_colors.get(component.lower(), Colors.WHITE)

        colored_component = self._colorize(f"[{component.upper()}]", component_color)

        task_part = ""
        if task_name:
            task_part = f" {self._colorize(f'[{task_name}]', Colors.BRIGHT_WHITE)}"

        return f"{timestamp} {colored_level} {colored_component}{task_part} {message}"

    def _write_log(self, message, to_stderr=False):
        stream = sys.stderr if to_stderr else self.output_stream
        stream.write(message + "\n")
        stream.flush()

    def debug(self, message, component="manager", task_name=None):
        formatted = self._format_message(LogLevel.DEBUG, component, message, task_name)
        self._write_log(formatted)

    def info(self, message, component="manager", task_name=None):
        formatted = self._format_message(LogLevel.INFO, component, message, task_name)
        self._write_log(formatted)

    def warning(self, message, component="manager", task_name=None):
        formatted = self._format_message(
            LogLevel.WARNING, component, message, task_name
        )
        self._write_log(formatted)

    def error(self, message, component="manager", task_name=None):
        formatted = self._format_message(LogLevel.ERROR, component, message, task_name)
        self._write_log(formatted, to_stderr=True)

    def critical(self, message, component="manager", task_name=None):
        formatted = self._format_message(
            LogLevel.CRITICAL, component, message, task_name
        )
        self._write_log(formatted, to_stderr=True)

    def task_started(self, task_name, component="task"):
        message = f"Task started: {self._colorize(task_name, Colors.BRIGHT_WHITE)}"
        self.info(message, component)

    def task_completed(self, task_name, component="task"):
        message = f"Task completed: {self._colorize(task_name, Colors.BRIGHT_WHITE)}"
        self.info(message, component)

    def task_killed(self, task_name, component="task"):
        message = f"Task killed: {self._colorize(task_name, Colors.BRIGHT_WHITE)}"
        self.warning(message, component)

    def manager_starting(self, task_count):
        message = (
            f"Starting with "
            f"{self._colorize(str(task_count), Colors.BRIGHT_WHITE)} "
            f"initial tasks"
        )
        self.info(message, "manager")

    def manager_exiting(self):
        self.info("All tasks finished. Exiting.", "manager")

    def task_log(self, message, level=LogLevel.INFO):
        task_component = f"TASK-{self.name.upper()}"
        formatted = self._format_message(level, task_component, message)
        self._write_log(
            formatted, to_stderr=level in [LogLevel.ERROR, LogLevel.CRITICAL]
        )

    def separator(self, title=None):
        if title:
            separator = f"{'=' * 20} {title} {'=' * 20}"
        else:
            separator = "=" * 50
        self._write_log(self._colorize(separator, Colors.BRIGHT_BLUE))
