import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import urllib.request
import zipfile
import io

# Ensure the current directory (GUI folder) is in sys.path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import concurrent.futures
import sqlite3
import h5py
import pandas as pd
from ultralytics import YOLO

import camera
import gui_components
import resource_paths

# List of required files relative to the resources folder.
REQUIRED_FILES = [
    os.path.join("run", "candidate_features.h5"),
    os.path.join("run", "card_database.db"),
    os.path.join("run", "faiss_ivf.index"),
    os.path.join("run", "index_to_card.txt"),
    os.path.join("yolo", "cardseg.pt"),
    os.path.join("yolo", "faceob.pt"),
    "scan_history.txt",
]

def resources_exist():
    """Check if all required resource files exist."""
    base = resource_paths.get_resource_base()
    for rel_path in REQUIRED_FILES:
        full_path = os.path.join(base, rel_path)
        if not os.path.exists(full_path):
            return False
    return True

class DownloadModal(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Downloading Resources")
        self.resizable(False, False)
        self.grab_set()  # Make modal.
        tk.Label(self, text="Downloading resources, please wait...").pack(padx=20, pady=10)
        self.progressbar = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=300)
        self.progressbar.pack(padx=20, pady=10)
        
        # Shared variables updated by the download thread.
        self.download_progress = 0  # percentage (0-100)
        self.download_complete_flag = False
        self.download_error = None
        self.lock = threading.Lock()
        
        # Start the download thread.
        threading.Thread(target=self.download_thread, daemon=True).start()
        # Start polling for progress.
        self.after(100, self.check_progress)
        
    def check_progress(self):
        """Periodically update the progress bar and check for completion."""
        with self.lock:
            progress = self.download_progress
            complete = self.download_complete_flag
            error = self.download_error
        self.progressbar['value'] = progress
        if complete:
            # Download thread finished; process result.
            if error is None:
                messagebox.showinfo("Download Complete", "Resources downloaded successfully.", parent=self)
            else:
                messagebox.showerror("Error", f"Failed to download resources: {error}", parent=self)
                self.master.destroy()
            self.destroy()
        else:
            # Continue polling.
            self.after(100, self.check_progress)
    
    def download_thread(self):
        url = "https://huggingface.co/datasets/JakeTurner616/mtg-cards-SIFT-Features/resolve/main/resources.zip?download=true"
        try:
            with urllib.request.urlopen(url) as response:
                total_length = response.getheader('Content-Length')
                if total_length is None:
                    data = response.read()
                    with self.lock:
                        self.download_progress = 100
                else:
                    total_length = int(total_length)
                    downloaded = 0
                    chunks = []
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        with self.lock:
                            self.download_progress = downloaded * 100 / total_length
                    data = b"".join(chunks)
            # Extract the ZIP.
            zip_data = io.BytesIO(data)
            with zipfile.ZipFile(zip_data) as zip_ref:
                # Extract into the parent directory so that the structure is preserved.
                resources_dir = resource_paths.get_resource_base()
                parent_dir = os.path.dirname(resources_dir)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                zip_ref.extractall(parent_dir)
            with self.lock:
                self.download_complete_flag = True
        except Exception as e:
            with self.lock:
                self.download_error = str(e)
                self.download_complete_flag = True

def check_and_download_resources(root):
    """Check for required files and force download if any are missing.
       Blocks further progress until resources exist or extraction fails."""
    if not resources_exist():
        modal = DownloadModal(root)
        # Wait until the modal window is closed (i.e. download completes).
        root.wait_window(modal)
        # Re-check resources.
        if not resources_exist():
            print("Debug: Some resource files are still missing after download.")
            root.destroy()
        else:
            print("Resources successfully verified after download.")
    else:
        print("All required resources are present.")

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
    # Force resources to be present before proceeding.
    check_and_download_resources(root)
    
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