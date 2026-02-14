#!/usr/bin/env python
"""
DSP Signal Processing Application
Main entry point for the GUI application.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main window
from gui.main_window import root

if __name__ == "__main__":
    # The GUI is already running via root.mainloop() in main_window.py
    # This file serves as the entry point
    pass
