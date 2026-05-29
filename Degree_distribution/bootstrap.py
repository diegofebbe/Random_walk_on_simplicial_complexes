import os, sys
"""
Simulate Pycharm's handling of source roots by adding the project root to sys.path.
If you use an IDE that handles source roots differently, you need this.
You can also ignore this, by changing the import statements.
"""

def ensure_root_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
