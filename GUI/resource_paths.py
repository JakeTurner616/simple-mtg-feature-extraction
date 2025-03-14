# GUI/resource_paths.py
import os
import sys

def get_resource_base():
    """
    Returns the absolute path to the resources folder.
    If running in a PyInstaller bundle, looks for the bundled 'resources' folder
    inside sys._MEIPASS.
    Otherwise, assumes resources is one folder up from the GUI directory.
    """
    if getattr(sys, 'frozen', False):
        # When frozen, resources should be bundled in sys._MEIPASS/resources.
        return os.path.join(sys._MEIPASS, "resources")
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

def resource_path(*paths):
    return os.path.join(get_resource_base(), *paths)