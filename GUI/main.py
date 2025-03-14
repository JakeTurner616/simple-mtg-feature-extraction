# GUI/main.py
import os
import sys
# Ensure the current directory (GUI folder) is in sys.path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tkinter as tk
import concurrent.futures
import sqlite3
import h5py
import pandas as pd
from ultralytics import YOLO

import camera
import gui_components
import resource_paths

def load_heavy_resources():
    # Camera selection and loading.
    selected = camera.select_camera_dialog(gui_components.root)
    gui_components.SELECTED_CAMERA_INDEX = selected
    gui_components.camera_reader = camera.load_camera(selected)
    if gui_components.camera_reader is None:
        print("Error: No camera detected.")
    # Load feature database and card information.
    gui_components.hf = h5py.File(resource_paths.resource_path("run", "candidate_features.h5"), 'r')
    conn = sqlite3.connect(resource_paths.resource_path("run", "card_database.db"))
    gui_components.label_mapping = pd.read_sql_query("SELECT * FROM cards", conn)
    conn.close()
    # Load FAISS index and index-to-card mapping.
    import faiss
    gui_components.faiss_index = faiss.read_index(resource_paths.resource_path("run", "faiss_ivf.index"))
    with open(resource_paths.resource_path("run", "index_to_card.txt"), 'r') as f:
        gui_components.index_to_card = [line.strip() for line in f]
    print("Heavy resources loaded.")

def main():
    root = tk.Tk()
    root.title("MTG Card Scanner")
    gui_components.root = root  # Make root available in gui_components.
    # Load segmentation and face models.
    gui_components.seg_model = YOLO(resource_paths.resource_path("yolo", "cardseg.pt"))
    gui_components.face_model = YOLO(resource_paths.resource_path("yolo", "faceob.pt"))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    future = executor.submit(load_heavy_resources)
    def on_resources_loaded(f):
        gui_components.setup_main_interface(root, executor)
    future.add_done_callback(lambda f: root.after(0, on_resources_loaded, f))
    root.protocol("WM_DELETE_WINDOW", lambda: gui_components.on_closing(root, executor))
    root.mainloop()

if __name__ == "__main__":
    main()
