# GUI/gui_components.py
import cv2
import time
import tkinter as tk
from tkinter import Label, Button, Text, Scrollbar, END, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import concurrent.futures
import pandas as pd
import sqlite3
import h5py
import os
import platform

import camera
import segmentation
import feature_extraction
import resource_paths

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Global configuration variables
MIN_SEG_CONFIDENCE = 0.95      
MIN_FACE_CONFIDENCE = 0.8     
STABLE_COUNT_THRESHOLD = 8    
DEBOUNCE_INTERVAL = 2.0       
CENTER_TOL = 40               
FRAME_FOCUS_THRESHOLD = 100.0  
WEIGHT_CENTERING = 1.0        
WEIGHT_FOCUS = 0.01           
WEIGHT_FACE = 1.0             
QUALITY_TRIGGER_THRESHOLD = 2.5  
ROI_CANDIDATE_COUNT = 3       
MIN_INLIER_THRESHOLD = 15     
MIN_ROI_SIZE = 50

# Global state and resource variables
history_file_path = resource_paths.resource_path("scan_history.txt")
roi_candidates = []
last_detected_card = None
seg_model = None
face_model = None
faiss_index = None
hf = None
label_mapping = None
index_to_card = None
camera_reader = None
SELECTED_CAMERA_INDEX = None

if platform.system() == "Windows":
    import winsound
    def beep():
        winsound.Beep(1000, 200)
else:
    def beep():
        print("\a")

def open_settings(root):
    settings_win = tk.Toplevel(root)
    settings_win.title("Settings")
    tk.Label(settings_win, text="Min Seg Confidence:").grid(row=0, column=0, sticky="e")
    seg_conf_entry = tk.Entry(settings_win)
    seg_conf_entry.grid(row=0, column=1)
    seg_conf_entry.insert(0, str(MIN_SEG_CONFIDENCE))
    tk.Label(settings_win, text="Min Face Confidence:").grid(row=1, column=0, sticky="e")
    face_conf_entry = tk.Entry(settings_win)
    face_conf_entry.grid(row=1, column=1)
    face_conf_entry.insert(0, str(MIN_FACE_CONFIDENCE))
    tk.Label(settings_win, text="Stable Count Threshold:").grid(row=2, column=0, sticky="e")
    stable_thresh_entry = tk.Entry(settings_win)
    stable_thresh_entry.grid(row=2, column=1)
    stable_thresh_entry.insert(0, str(STABLE_COUNT_THRESHOLD))
    tk.Label(settings_win, text="Debounce Interval (s):").grid(row=3, column=0, sticky="e")
    debounce_entry = tk.Entry(settings_win)
    debounce_entry.grid(row=3, column=1)
    debounce_entry.insert(0, str(DEBOUNCE_INTERVAL))
    tk.Label(settings_win, text="Min ROI Size (px):").grid(row=4, column=0, sticky="e")
    min_roi_entry = tk.Entry(settings_win)
    min_roi_entry.grid(row=4, column=1)
    min_roi_entry.insert(0, str(MIN_ROI_SIZE))
    tk.Label(settings_win, text="History File Path:").grid(row=5, column=0, sticky="e")
    hist_path_entry = tk.Entry(settings_win, width=40)
    hist_path_entry.grid(row=5, column=1)
    hist_path_entry.insert(0, history_file_path)
    def save_settings():
        try:
            global MIN_SEG_CONFIDENCE, MIN_FACE_CONFIDENCE, STABLE_COUNT_THRESHOLD, DEBOUNCE_INTERVAL, MIN_ROI_SIZE, history_file_path
            MIN_SEG_CONFIDENCE = float(seg_conf_entry.get())
            MIN_FACE_CONFIDENCE = float(face_conf_entry.get())
            STABLE_COUNT_THRESHOLD = int(stable_thresh_entry.get())
            DEBOUNCE_INTERVAL = float(debounce_entry.get())
            MIN_ROI_SIZE = int(min_roi_entry.get())
            history_file_path = hist_path_entry.get()
            settings_win.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
    Button(settings_win, text="Save Settings", command=save_settings).grid(row=6, column=0, columnspan=2, pady=10)

def start_scanner(root, executor):
    process_frame(root, executor)

# Global state variables for the processing loop
seg_future = None
frame_counter = 0
last_roi_capture_time = 0
feature_future = None
inference_in_progress = False
state = "SEARCHING"
stable_count = 0
quality_sum = 0.0
last_valid_quad = None
progress_var = None
history_text = None
status_label = None
detected_card_label = None
inference_label = None
roi_label = None

def process_frame(root, executor):
    global camera_reader, seg_future, frame_counter, last_roi_capture_time, feature_future, inference_in_progress
    global state, stable_count, quality_sum, last_valid_quad, roi_candidates, last_detected_card
    if camera_reader is None:
        print("Camera reader is None. Attempting to re-open camera...")
        camera_reader = camera.load_camera(SELECTED_CAMERA_INDEX)
        if camera_reader is None:
            print("Failed to re-open camera. Please check the camera index.")
            root.after(1000, lambda: process_frame(root, executor))
            return
    ret, frame = camera_reader.read()
    if not ret:
        print("Failed to read frame from camera.")
        root.after(15, lambda: process_frame(root, executor))
        return
    frame_counter += 1
    status_text = f"State: {state} | "
    frame_focus = segmentation.compute_focus_measure(frame)
    if frame_focus < FRAME_FOCUS_THRESHOLD:
        status_text += f"Frame blurry ({frame_focus:.2f}); skipping processing."
        state = "SEARCHING"
        stable_count = 0
        quality_sum = 0.0
        roi_candidates = []
        progress_var.set(0)
        status_label.config(text=status_text)
    else:
        if frame_counter % 5 == 0:
            seg_future = executor.submit(segmentation.process_segmentation, frame.copy(), seg_model)
        if seg_future is not None and seg_future.done():
            quad = seg_future.result()
            if quad is not None:
                last_valid_quad = quad
        detected_quad = last_valid_quad
        if detected_quad is not None:
            cv2.polylines(frame, [detected_quad], True, (0, 255, 0), 3)
            centered = segmentation.is_card_centered(detected_quad, frame, tol=CENTER_TOL)
            roi_candidate = segmentation.four_point_transform(frame, detected_quad)
            if roi_candidate.shape[0] < MIN_ROI_SIZE or roi_candidate.shape[1] < MIN_ROI_SIZE:
                status_text += "ROI too small; ignoring. "
                state = "SEARCHING"
                stable_count = 0
                quality_sum = 0.0
                roi_candidates = []
                progress_var.set(0)
            else:
                focus_measure = segmentation.compute_focus_measure(roi_candidate)
                face_detected = feature_extraction.detect_face_elements(roi_candidate, face_model)
                quality_score = (WEIGHT_CENTERING if centered else 0.0) + (WEIGHT_FOCUS * focus_measure) + (WEIGHT_FACE if face_detected else 0.0)
                status_text += f"Centered: {centered}, Focus: {focus_measure:.2f}, Face: {face_detected}, Score: {quality_score:.2f} | "
                if state == "SEARCHING":
                    if quality_score >= (QUALITY_TRIGGER_THRESHOLD * 0.8):
                        state = "CANDIDATE"
                        stable_count = 1
                        quality_sum = quality_score
                elif state == "CANDIDATE":
                    if quality_score >= (QUALITY_TRIGGER_THRESHOLD * 0.8):
                        stable_count += 1
                        quality_sum += quality_score
                        if stable_count >= (STABLE_COUNT_THRESHOLD // 2):
                            state = "STABLE"
                    else:
                        state = "SEARCHING"
                        stable_count = 0
                        quality_sum = 0.0
                elif state == "STABLE":
                    if quality_score >= (QUALITY_TRIGGER_THRESHOLD * 0.8):
                        stable_count += 1
                        quality_sum += quality_score
                        avg_quality = quality_sum / stable_count
                        roi_candidates.append((roi_candidate, avg_quality))
                        status_text += f"Candidate ROI added (avg quality: {avg_quality:.2f}). Total: {len(roi_candidates)} | "
                        current_time = time.time()
                        if (current_time - last_roi_capture_time) >= DEBOUNCE_INTERVAL:
                            state = "TRIGGERING"
                            roi_image = cv2.cvtColor(max(roi_candidates, key=lambda x: x[1])[0], cv2.COLOR_BGR2RGB)
                            roi_image = Image.fromarray(roi_image)
                            roi_image.thumbnail((356, 356), Image.LANCZOS)
                            roi_photo = ImageTk.PhotoImage(roi_image)
                            roi_label.config(image=roi_photo)
                            roi_label.image = roi_photo
                            status_text += " | ROI Captured!"
                            last_roi_capture_time = current_time
                            roi_candidates = []
                            inference_in_progress = True
                            inference_label.config(text="Inference: Running...")
                            beep()
                            inf_start = time.perf_counter()
                            feature_future = executor.submit(feature_extraction.find_closest_card_ransac, roi_candidate, faiss_index, index_to_card, hf, label_mapping, 5)
                            inf_end = time.perf_counter()
                            print(f"Triggering inference took {inf_end - inf_start:.3f} seconds")
                    else:
                        state = "SEARCHING"
                        stable_count = 0
                        quality_sum = 0.0
                        roi_candidates = []
                progress_var.set(stable_count)
                status_text += f"Stable Count: {stable_count}"
        else:
            state = "SEARCHING"
            stable_count = 0
            quality_sum = 0.0
            roi_candidates = []
            progress_var.set(0)
            status_text += "No detection."
    if feature_future is not None and feature_future.done():
        best_candidate, card_name, kps, proc_img, debug_info = feature_future.result()
        detected_card_label.config(text=f"Detected Card: {card_name}")
        if card_name != "Unknown":
            history_text.insert(END, f"{card_name}\n")
            history_text.see(END)
            try:
                with open(history_file_path, "a") as f:
                    f.write(f"{card_name}\n")
            except Exception as e:
                print("Error writing history to file:", e)
        last_detected_card = best_candidate
        inference_label.config(text="Inference: Finished")
        inference_in_progress = False
        feature_future = None
        state = "SEARCHING"
        stable_count = 0
        quality_sum = 0.0
    status_label.config(text=status_text)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((456, 456), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.config(image=imgtk)
    video_label.image = imgtk
    root.after(15, lambda: process_frame(root, executor))

def on_closing(root, executor):
    global camera_reader, hf
    try:
        if camera_reader is not None:
            camera_reader.release()
    except Exception:
        pass
    executor.shutdown(wait=False)
    try:
        hf.close()
    except Exception:
        pass
    root.destroy()

def setup_main_interface(root, executor):
    # Clear existing widgets.
    for widget in root.winfo_children():
        widget.destroy()

    # Create settings button.
    Button(root, text="Settings", command=lambda: open_settings(root)).grid(row=2, column=0, sticky="w", padx=10, pady=5)

    global camera_frame, status_frame, history_frame, video_label, roi_label, status_label, detected_card_label, progress_var, inference_label, history_text
    camera_frame = tk.Frame(root)
    camera_frame.grid(row=0, column=0, padx=10, pady=10)
    from PIL import Image
    placeholder_camera = Image.new("RGB", (456, 456), (150, 150, 150))
    placeholder_camera_photo = ImageTk.PhotoImage(placeholder_camera)
    video_label = Label(camera_frame, image=placeholder_camera_photo)
    video_label.image = placeholder_camera_photo
    video_label.grid(row=0, column=0, padx=10, pady=10)

    placeholder_image = Image.new("RGB", (356, 356), (200, 200, 200))
    placeholder_photo = ImageTk.PhotoImage(placeholder_image)
    roi_label = Label(camera_frame, image=placeholder_photo)
    roi_label.image = placeholder_photo
    roi_label.grid(row=0, column=1, padx=10, pady=10)

    status_frame = tk.Frame(root)
    status_frame.grid(row=1, column=0, padx=10, pady=10)
    status_label = Label(status_frame, text="Initializing...", font=("Helvetica", 12))
    status_label.grid(row=0, column=0, columnspan=2)
    detected_card_label = Label(status_frame, text="Detected Card: None", font=("Helvetica", 14))
    detected_card_label.grid(row=1, column=0, columnspan=2, pady=5)
    progress_var = tk.DoubleVar()
    ttk.Progressbar(status_frame, variable=progress_var, maximum=STABLE_COUNT_THRESHOLD, length=300).grid(row=2, column=0, columnspan=2, pady=5)
    inference_label = Label(status_frame, text="Inference: Idle", font=("Helvetica", 12))
    inference_label.grid(row=3, column=0, columnspan=2, pady=5)

    history_frame = tk.Frame(root)
    history_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
    Label(history_frame, text="Scanned Cards History", font=("Helvetica", 14)).grid(row=0, column=0, pady=5)
    history_text = Text(history_frame, height=20, width=40)
    history_text.grid(row=1, column=0, pady=5)
    scrollbar = Scrollbar(history_frame, orient="vertical", command=lambda *args: history_text.yview(*args))
    scrollbar.grid(row=1, column=1, sticky='ns')
    history_text.config(yscrollcommand=scrollbar.set)

    Button(root, text="Exit", command=lambda: on_closing(root, executor)).grid(row=2, column=1, pady=5)

    # Initialize loop state variables.
    global seg_future, frame_counter, last_roi_capture_time, feature_future, inference_in_progress, state, stable_count, quality_sum, last_valid_quad
    seg_future = None
    frame_counter = 0
    last_roi_capture_time = 0
    feature_future = None
    inference_in_progress = False
    state = "SEARCHING"
    stable_count = 0
    quality_sum = 0.0
    last_valid_quad = None

    start_scanner(root, executor)
