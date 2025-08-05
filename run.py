#!/usr/bin/env python3
"""
Glenn Bot - AI Collaboration Assistant
Run this script to start the bot.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")