import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import Label, Button, Text, Scrollbar, VERTICAL, END, messagebox
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import concurrent.futures
import json
import pandas as pd
import faiss
import h5py
import sqlite3
from collections import Counter
import os
import platform
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Define a beep function
if platform.system() == "Windows":
    import winsound
    def beep():
        winsound.Beep(1000, 200)
else:
    def beep():
        print("\a")

##############################
# Configuration for Thresholds (global variables)
##############################
MIN_SEG_CONFIDENCE = 0.95      
MIN_FACE_CONFIDENCE = 0.8     
STABLE_COUNT_THRESHOLD = 8    
ROI_FOCUS_THRESHOLD = 100.0   
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

# File path to save scan history (default)
history_file_path = "resources/scan_history.txt"

##############################
# Global Variables for Candidate Management
##############################
roi_candidates = []           
last_detected_card = None     

##############################
# Global Variables for Heavy Resources and Camera Selection
##############################
seg_model = None
face_model = None
faiss_index = None
hf = None
label_mapping = None
index_to_card = None
camera_reader = None   # OpenCV VideoCapture object
SELECTED_CAMERA_INDEX = None

##############################
# Camera Functions
##############################
def list_available_cameras(max_indices=5):
    available = []
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available

def select_camera_dialog(root):
    global SELECTED_CAMERA_INDEX
    available_cams = list_available_cameras()
    if not available_cams:
        messagebox.showerror("Error", "No cameras found!")
        root.destroy()
        return

    dialog = tk.Toplevel(root)
    dialog.title("Select Camera")
    tk.Label(dialog, text="Select a camera from the list:").grid(row=0, column=0, padx=10, pady=10)
    
    cam_var = tk.StringVar()
    cam_combobox = ttk.Combobox(dialog, textvariable=cam_var, state="readonly")
    cam_combobox['values'] = available_cams
    cam_combobox.current(0)
    cam_combobox.grid(row=1, column=0, padx=10, pady=10)

    def confirm_selection():
        try:
            SELECTED_CAMERA_INDEX = int(cam_var.get())
            globals()['SELECTED_CAMERA_INDEX'] = SELECTED_CAMERA_INDEX
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Invalid Selection", "Please select a valid camera index.")

    confirm_button = Button(dialog, text="Confirm", command=confirm_selection)
    confirm_button.grid(row=2, column=0, padx=10, pady=10)
    
    dialog.grab_set()  # Make the dialog modal
    root.wait_window(dialog)

def load_camera():
    global SELECTED_CAMERA_INDEX
    if SELECTED_CAMERA_INDEX is None:
        print("Error: No camera selected.")
        return None
    cap = cv2.VideoCapture(SELECTED_CAMERA_INDEX)
    if cap.isOpened():
        print(f"Camera {SELECTED_CAMERA_INDEX} opened!")
        return cap
    print(f"Error: Unable to open camera {SELECTED_CAMERA_INDEX}.")
    return None

##############################
# Helper Functions for Segmentation & ROI Extraction
##############################
def approximate_quad(contour, desired_points=4, initial_scale=0.01, step_scale=0.005, max_scale=0.1):
    arc_len = cv2.arcLength(contour, True)
    scale = initial_scale
    approx = cv2.approxPolyDP(contour, scale * arc_len, True)
    while len(approx) != desired_points and scale < max_scale:
        scale += step_scale
        approx = cv2.approxPolyDP(contour, scale * arc_len, True)
    return approx

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def compute_focus_measure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_segmentation(frame):
    start_time = time.perf_counter()
    try:
        results = seg_model(frame, conf=0.7)
        result = results[0]
        if result.masks is not None and result.masks.xy is not None:
            for i, polygon in enumerate(result.masks.xy):
                confidence = result.masks.conf[i] if hasattr(result.masks, 'conf') else 1.0
                if confidence < MIN_SEG_CONFIDENCE:
                    continue
                contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                quad = approximate_quad(contour)
                if quad is None or len(quad) != 4:
                    rect = cv2.minAreaRect(contour)
                    quad = cv2.boxPoints(rect)
                    quad = quad.astype(np.int32)
                else:
                    quad = quad.reshape(-1, 2).astype(np.int32)
                end_time = time.perf_counter()
                print(f"Segmentation step took {end_time - start_time:.3f} seconds")
                return quad  
    except Exception as e:
        print("Segmentation error:", e)
    end_time = time.perf_counter()
    print(f"Segmentation step took {end_time - start_time:.3f} seconds (no valid detection)")
    return None

def is_card_centered(quad, frame, tol=CENTER_TOL):
    if quad is None:
        return False
    M = cv2.moments(quad)
    if M["m00"] == 0:
        return False
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    (h, w) = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    return dist <= tol

##############################
# Helper Functions for Feature Extraction & Matching
##############################
def load_candidate_features_for_card(card_id, hf):
    features = []
    if card_id in hf:
        card_grp = hf[card_id]
        for feat_key in card_grp.keys():
            kp_json_arr = card_grp[feat_key]["keypoints"][()]
            kp_str = kp_json_arr[0].decode("utf-8") if isinstance(kp_json_arr[0], bytes) else kp_json_arr[0]
            kp_serialized = json.loads(kp_str)
            des = card_grp[feat_key]["descriptors"][()].astype('float32')
            features.append((kp_serialized, des))
    return features

def deserialize_keypoints(kps_data):
    keypoints = []
    for d in kps_data:
        kp = cv2.KeyPoint(d['pt'][0], d['pt'][1],
                          d['size'], d['angle'],
                          d['response'], d['octave'],
                          d['class_id'])
        keypoints.append(kp)
    return keypoints

def extract_features_sift(roi_image, max_features=100):
    start_time = time.perf_counter()
    resized = cv2.resize(roi_image, (256, 256))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))
    enhanced_color = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is not None and len(keypoints) > max_features:
        sorted_kp_des = sorted(zip(keypoints, descriptors), key=lambda x: -x[0].response)
        keypoints, descriptors = zip(*sorted_kp_des[:max_features])
        keypoints, descriptors = list(keypoints), np.array(descriptors)
    
    if descriptors is not None:
        eps = 1e-7
        descriptors = descriptors / (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        descriptors = descriptors.astype('float16')
    
    end_time = time.perf_counter()
    print(f"Feature extraction (SIFT) took {end_time - start_time:.3f} seconds")
    return keypoints, descriptors, enhanced_color

def find_closest_card_ransac(roi_image, faiss_index, index_to_card, hf, label_mapping, k=5, min_candidate_matches=2):
    overall_start = time.perf_counter()
    debug_info = {}

    start_feat = time.perf_counter()
    keypoints, descriptors, processed_img = extract_features_sift(roi_image, max_features=100)
    debug_info['num_keypoints'] = len(keypoints) if keypoints else 0
    feat_time = time.perf_counter() - start_feat
    print(f"Total SIFT extraction time: {feat_time:.3f} seconds")

    if descriptors is None or len(keypoints) == 0:
        debug_info['error'] = "No descriptors found."
        overall_end = time.perf_counter()
        print(f"Inference step took {overall_end - overall_start:.3f} seconds")
        return None, "Unknown", keypoints, processed_img, debug_info

    descriptors = descriptors.astype('float32')
    start_faiss = time.perf_counter()
    distances, indices = faiss_index.search(descriptors, k)
    candidate_ids = [index_to_card[i] for i in indices.flatten()]
    candidate_counts = Counter(candidate_ids)
    debug_info['faiss_candidate_counts'] = dict(candidate_counts)
    faiss_time = time.perf_counter() - start_faiss
    print(f"FAISS search time: {faiss_time:.3f} seconds")

    bf = cv2.BFMatcher()
    best_inliers = 0
    best_candidate = None

    start_ransac = time.perf_counter()
    for candidate_id, count in candidate_counts.items():
        if count < min_candidate_matches:
            continue
        candidate_sets = load_candidate_features_for_card(candidate_id, hf)
        total_inliers = 0
        for kp_serialized, candidate_des in candidate_sets:
            candidate_kp = deserialize_keypoints(kp_serialized)
            matches = bf.knnMatch(descriptors, candidate_des.astype('float32'), k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good_matches) >= 4:
                src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([candidate_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    total_inliers += inliers
        debug_info.setdefault('ransac_inliers', {})[candidate_id] = total_inliers
        if total_inliers > best_inliers:
            best_inliers = total_inliers
            best_candidate = candidate_id

    ransac_time = time.perf_counter() - start_ransac
    print(f"RANSAC matching time: {ransac_time:.3f} seconds")
    debug_info['best_inliers'] = best_inliers

    if best_inliers < MIN_INLIER_THRESHOLD:
        best_candidate = None

    try:
        if best_candidate is not None:
            card_row = label_mapping.loc[label_mapping['scryfall_id'] == best_candidate]
            card_name = card_row['name'].values[0] if not card_row.empty else "Unknown"
        else:
            card_name = "Unknown"
    except Exception:
        card_name = "Unknown"

    overall_end = time.perf_counter()
    print(f"Inference step took {overall_end - overall_start:.3f} seconds")
    return best_candidate, card_name, keypoints, processed_img, debug_info

def detect_face_elements(roi_image):
    results = face_model(roi_image, conf=0.8)
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            conf = box.conf.item() if hasattr(box, 'conf') else 1.0
            if conf >= MIN_FACE_CONFIDENCE:
                return True
    return False

##############################
# Heavy Resource Loading in Background
##############################
def load_heavy_resources():
    global seg_model, face_model, faiss_index, hf, label_mapping, index_to_card, camera_reader
    seg_model = YOLO("resources/yolo/cardseg.pt")
    face_model = YOLO("resources/yolo/faceob.pt")
    # Load the camera using the user-selected camera index
    camera_reader = load_camera()
    if camera_reader is None:
        print("Error: No camera detected with the selected index.")
    hf = h5py.File('resources/run/candidate_features.h5', 'r')
    conn = sqlite3.connect('resources/run/card_database.db')
    label_mapping = pd.read_sql_query("SELECT * FROM cards", conn)
    conn.close()
    faiss_index = faiss.read_index("resources/run/faiss_ivf.index")
    with open('resources/run/index_to_card.txt', 'r') as f:
        index_to_card = [line.strip() for line in f]
    print("Heavy resources loaded.")

##############################
# Tkinter GUI Setup (Initial Loader)
##############################
root = tk.Tk()
root.title("MTG Card Scanner")

# First, ask the user to select a camera.
select_camera_dialog(root)

loading_label = Label(root, text="Loading Segmentation model into memory...", font=("Helvetica", 16))
loading_label.grid(row=0, column=0, padx=20, pady=20)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
future = executor.submit(load_heavy_resources)

def on_resources_loaded(future):
    loading_label.config(text="Scanner Ready!")
    root.after(500, setup_main_interface)

future.add_done_callback(lambda f: root.after(0, on_resources_loaded, f))

##############################
# Settings Window
##############################
def open_settings():
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
        global MIN_SEG_CONFIDENCE, MIN_FACE_CONFIDENCE, STABLE_COUNT_THRESHOLD, DEBOUNCE_INTERVAL, MIN_ROI_SIZE, history_file_path
        try:
            MIN_SEG_CONFIDENCE = float(seg_conf_entry.get())
            MIN_FACE_CONFIDENCE = float(face_conf_entry.get())
            STABLE_COUNT_THRESHOLD = int(stable_thresh_entry.get())
            DEBOUNCE_INTERVAL = float(debounce_entry.get())
            MIN_ROI_SIZE = int(min_roi_entry.get())
            history_file_path = hist_path_entry.get()
            settings_win.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

    save_button = Button(settings_win, text="Save Settings", command=save_settings)
    save_button.grid(row=6, column=0, columnspan=2, pady=10)

##############################
# Main Interface Setup and Processing Loop
##############################
def setup_main_interface():
    loading_label.grid_forget()
    
    settings_button = Button(root, text="Settings", command=open_settings)
    settings_button.grid(row=2, column=0, sticky="w", padx=10, pady=5)
    
    global camera_frame, status_frame, history_frame, video_label, roi_label
    camera_frame = tk.Frame(root)
    camera_frame.grid(row=0, column=0, padx=10, pady=10)
    
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
    
    global status_label, detected_card_label, progress_var, inference_label
    status_label = Label(status_frame, text="Initializing...", font=("Helvetica", 12))
    status_label.grid(row=0, column=0, columnspan=2)
    
    detected_card_label = Label(status_frame, text="Detected Card: None", font=("Helvetica", 14))
    detected_card_label.grid(row=1, column=0, columnspan=2, pady=5)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(status_frame, variable=progress_var, maximum=STABLE_COUNT_THRESHOLD, length=300)
    progress_bar.grid(row=2, column=0, columnspan=2, pady=5)
    
    inference_label = Label(status_frame, text="Inference: Idle", font=("Helvetica", 12))
    inference_label.grid(row=3, column=0, columnspan=2, pady=5)
    
    history_frame = tk.Frame(root)
    history_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)
    
    history_label = Label(history_frame, text="Scanned Cards History", font=("Helvetica", 14))
    history_label.grid(row=0, column=0, pady=5)
    
    global history_text
    history_text = Text(history_frame, height=20, width=40)
    history_text.grid(row=1, column=0, pady=5)
    
    scrollbar = Scrollbar(history_frame, orient=VERTICAL)
    scrollbar.grid(row=1, column=1, sticky='ns')
    history_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=history_text.yview)
    
    exit_button = Button(root, text="Exit", command=on_closing)
    exit_button.grid(row=2, column=1, pady=5)
    
    global seg_future, frame_counter, last_roi_capture_time, feature_future, inference_in_progress
    global state, stable_count, quality_sum, last_valid_quad
    seg_future = None
    frame_counter = 0
    last_roi_capture_time = 0
    feature_future = None
    inference_in_progress = False
    state = "SEARCHING"
    stable_count = 0
    quality_sum = 0.0
    last_valid_quad = None
    
    start_scanner()

def process_frame():
    global frame_counter, last_roi_capture_time, feature_future, inference_in_progress
    global seg_future, state, stable_count, quality_sum, roi_candidates, last_valid_quad, last_detected_card, camera_reader

    # Check if camera_reader is valid; if not, try to re-open it.
    if camera_reader is None:
        print("Camera reader is None. Attempting to re-open camera...")
        camera_reader = load_camera()
        if camera_reader is None:
            print("Failed to re-open camera. Please check the camera index.")
            root.after(1000, process_frame)
            return

    ret, frame = camera_reader.read()
    if not ret:
        print("Failed to read frame from camera.")
        root.after(15, process_frame)
        return

    frame_counter += 1
    status_text = f"State: {state} | "
    frame_focus = compute_focus_measure(frame)
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
            seg_future = executor.submit(process_segmentation, frame.copy())
        if seg_future is not None and seg_future.done():
            quad = seg_future.result()
            if quad is not None:
                last_valid_quad = quad
        detected_quad = last_valid_quad
        if detected_quad is not None:
            cv2.polylines(frame, [detected_quad], isClosed=True, color=(0, 255, 0), thickness=3)
            centered = is_card_centered(detected_quad, frame, tol=CENTER_TOL)
            roi_candidate = four_point_transform(frame, detected_quad)
            if roi_candidate.shape[0] < MIN_ROI_SIZE or roi_candidate.shape[1] < MIN_ROI_SIZE:
                status_text += "ROI too small; ignoring. "
                state = "SEARCHING"
                stable_count = 0
                quality_sum = 0.0
                roi_candidates = []
                progress_var.set(0)
            else:
                focus_measure = compute_focus_measure(roi_candidate)
                face_detected = detect_face_elements(roi_candidate)
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
                            feature_future = executor.submit(find_closest_card_ransac, roi_candidate, faiss_index, index_to_card, hf, label_mapping, 5)
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

    root.after(15, process_frame)

def start_scanner():
    status_label.config(text="Scanner Running...")
    process_frame()

def on_closing():
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

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
