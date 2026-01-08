import os, sys
"""
Adjust for Pycharm python console handling of source roots.
If you use an IDE that handles source roots differently, 
you don't need this here, but just on the subdirectories.
"""

def ensure_root_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
