"""
CELLEX CANCER DETECTION SYSTEM - PROFESSIONAL LOGGER
===================================================
Advanced logging system with colors and professional formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback when colorama is not available
    class MockColor:
        RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = WHITE = ""
        RESET = BRIGHT = ""
    Fore = Back = Style = MockColor()

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class CellexFormatter(logging.Formatter):
    """Custom formatter for Cellex logging with colors and professional styling."""
    
    def __init__(self):
        super().__init__()
        
        # Define colors for different log levels
        self.colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
        }
        
        # Professional log format
        self.format_string = (
            f"{Fore.BLUE}[%(asctime)s]{Style.RESET_ALL} "
            f"{Fore.MAGENTA}[CELLEX]{Style.RESET_ALL} "
            f"%(levelname_colored)s "
            f"{Fore.WHITE}%(name)s{Style.RESET_ALL} - "
            f"%(message)s"
        )
    
    def format(self, record):
        # Add colored level name
        level_color = self.colors.get(record.levelname, Fore.WHITE)
        record.levelname_colored = f"{level_color}[{record.levelname:8}]{Style.RESET_ALL}"
        
        # Create formatter with our format string
        formatter = logging.Formatter(
            self.format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        return formatter.format(record)


class CellexLogger:
    """Professional logging system for Cellex AI."""
    
    def __init__(self, name: str = "Cellex", log_file: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
            
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CellexFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            self._add_file_handler(log_file)
    
    def _add_file_handler(self, log_file: str):
        """Add file handler for logging to file."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # File formatter without colors
        file_formatter = logging.Formatter(
            '[%(asctime)s] [CELLEX] [%(levelname)-8s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def success(self, message: str):
        """Log success message (custom level)."""
        self.info(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    
    def section(self, title: str):
        """Log section header."""
        separator = "=" * 60
        self.info(f"\n{Fore.CYAN}{separator}")
        self.info(f"{Fore.CYAN}  {title.upper()}")
        self.info(f"{Fore.CYAN}{separator}{Style.RESET_ALL}")
    
    def subsection(self, title: str):
        """Log subsection header."""
        separator = "-" * 40
        self.info(f"\n{Fore.BLUE}{separator}")
        self.info(f"{Fore.BLUE}  {title}")
        self.info(f"{Fore.BLUE}{separator}{Style.RESET_ALL}")
    
    def metric(self, name: str, value: float, unit: str = ""):
        """Log a metric with formatting."""
        self.info(f"{Fore.YELLOW}📊 {name}: {Fore.WHITE}{value:.4f} {unit}{Style.RESET_ALL}")
    
    def progress(self, current: int, total: int, description: str = ""):
        """Log progress information."""
        percentage = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        self.info(
            f"{Fore.CYAN}🔄 {description} "
            f"{Fore.WHITE}[{bar}] "
            f"{Fore.GREEN}{percentage:5.1f}% "
            f"{Fore.WHITE}({current}/{total}){Style.RESET_ALL}"
        )
    
    def step(self, step_num: int, total_steps: int, description: str):
        """Log training/processing step."""
        self.info(
            f"{Fore.MAGENTA}Step {step_num:3d}/{total_steps} - "
            f"{Fore.WHITE}{description}{Style.RESET_ALL}"
        )
    
    def model_info(self, model_name: str, params: int):
        """Log model information."""
        params_str = f"{params:,}" if params < 1_000_000 else f"{params/1_000_000:.1f}M"
        self.info(
            f"{Fore.BLUE}🧠 Model: {Fore.WHITE}{model_name} "
            f"{Fore.BLUE}| Parameters: {Fore.WHITE}{params_str}{Style.RESET_ALL}"
        )
    
    def training_epoch(self, epoch: int, total_epochs: int, train_loss: float, 
                      val_loss: float, accuracy: float):
        """Log training epoch results."""
        self.info(
            f"{Fore.GREEN}Epoch {epoch:3d}/{total_epochs} - "
            f"{Fore.WHITE}Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.2%}{Style.RESET_ALL}"
        )
    
    def banner(self, text: str):
        """Display a professional banner."""
        width = max(60, len(text) + 10)
        border = "═" * width
        
        print(f"\n{Fore.CYAN}{border}")
        print(f"{Fore.CYAN}║{' ' * ((width - len(text)) // 2)}{Fore.WHITE}{Style.BRIGHT}{text}{Style.RESET_ALL}{Fore.CYAN}{' ' * ((width - len(text)) // 2)}║")
        print(f"{Fore.CYAN}{border}{Style.RESET_ALL}\n")
    
    def welcome(self):
        """Display welcome banner for Cellex."""
        self.banner("CELLEX CANCER DETECTION SYSTEM")
        self.info(f"{Fore.WHITE}🏥 Advanced AI-Powered Medical Imaging Analysis")
        self.info(f"{Fore.WHITE}🔬 Built by AI Scientists for Medical Professionals")
        self.info(f"{Fore.WHITE}📅 Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


# Global logger instance
logger = CellexLogger()


def get_logger(name: str = "Cellex", log_file: Optional[str] = None) -> CellexLogger:
    """Get a Cellex logger instance."""
    return CellexLogger(name, log_file)


if __name__ == "__main__":
    # Demo of the logging system
    demo_logger = get_logger("Demo")
    
    demo_logger.welcome()
    demo_logger.section("SYSTEM INITIALIZATION")
    demo_logger.info("Loading configuration...")
    demo_logger.success("Configuration loaded successfully")
    
    demo_logger.subsection("Model Loading")
    demo_logger.model_info("EfficientNet-B0", 5_288_548)
    
    demo_logger.section("TRAINING PROGRESS")
    for epoch in range(1, 4):
        demo_logger.training_epoch(epoch, 3, 0.045 - epoch*0.01, 0.052 - epoch*0.01, 0.92 + epoch*0.02)
        demo_logger.progress(epoch, 3, "Training Progress")
    
    demo_logger.success("Training completed successfully!")
    demo_logger.metric("Final Accuracy", 0.9654, "%")