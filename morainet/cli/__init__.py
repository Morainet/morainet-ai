"""Morainet CLI — command-line management tools.

Usage::

    python -m morainet.cli run "What is AI?"            # single agent run
    python -m morainet.cli batch queries.txt            # batch execute
    python -m morainet.cli trace export results/        # export traces
    python -m morainet.cli memory clean --store redis   # clean memory
    python -m morainet.cli tool schema my_tool.py       # debug tool schema
    python -m morainet.cli workflow viz my_wf.py        # visualize workflow
"""
