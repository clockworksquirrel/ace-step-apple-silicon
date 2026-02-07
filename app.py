"""
HF Spaces / Quick Launch entry point for ACE-Step 1.5.

Usage:
    python app.py                     # Launch Gradio UI on localhost:7860
    python app.py --share             # Create public link
    python app.py --server-name 0.0.0.0  # Listen on all interfaces

This is a thin wrapper around the full pipeline — all CLI arguments
from acestep_v15_pipeline are supported.
"""

import sys

from acestep.acestep_v15_pipeline import main

if __name__ == "__main__":
    main()
