# GUI/camera.py
import cv2
import tkinter as tk
from tkinter import messagebox, ttk, Label, Button

def list_available_cameras(max_indices=5):
    available = []
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available

def select_camera_dialog(root):
    """
    Displays a dialog for selecting a camera index.
    Returns the selected index or None if no camera is found.
    """
    available_cams = list_available_cameras()
    if not available_cams:
        messagebox.showerror("Error", "No cameras found!")
        root.destroy()
        return None

    dialog = tk.Toplevel(root)
    dialog.title("Select Camera")
    Label(dialog, text="Select a camera from the list:").grid(row=0, column=0, padx=10, pady=10)

    cam_var = tk.StringVar()
    combobox = ttk.Combobox(dialog, textvariable=cam_var, state="readonly")
    combobox['values'] = available_cams
    combobox.current(0)
    combobox.grid(row=1, column=0, padx=10, pady=10)

    selected = {}

    def confirm_selection():
        try:
            selected['index'] = int(cam_var.get())
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Invalid Selection", "Please select a valid camera index.")

    Button(dialog, text="Confirm", command=confirm_selection).grid(row=2, column=0, padx=10, pady=10)
    dialog.grab_set()  # Make the dialog modal
    root.wait_window(dialog)
    return selected.get('index')

def load_camera(selected_index):
    """
    Attempts to open the camera at the given index.
    Returns a VideoCapture object or None if unsuccessful.
    """
    if selected_index is None:
        print("Error: No camera selected.")
        return None
    cap = cv2.VideoCapture(selected_index)
    if cap.isOpened():
        print(f"Camera {selected_index} opened!")
        return cap
    print(f"Error: Unable to open camera {selected_index}.")
    return None