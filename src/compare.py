"""
Legacy compatibility entry point.

Model comparison and optimization are now handled by optimizer.py.
The complete execution pipeline is run_pipeline.py.
"""

from run_pipeline import main


if __name__ == "__main__":
    main()