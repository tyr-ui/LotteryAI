"""
LotteryAI compatibility entry point.

The application pipeline is implemented in run_pipeline.py.
Data loading, feature calculation, prediction, backtesting, and
optimization are implemented in their respective modules.
"""

from run_pipeline import main


if __name__ == "__main__":
    main()