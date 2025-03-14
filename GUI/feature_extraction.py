# GUI/feature_extraction.py
import cv2
import numpy as np
import time
import json
from collections import Counter

MIN_INLIER_THRESHOLD = 15
MIN_FACE_CONFIDENCE = 0.8

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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

def detect_face_elements(roi_image, face_model):
    results = face_model(roi_image, conf=0.8)
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            conf = box.conf.item() if hasattr(box, 'conf') else 1.0
            if conf >= MIN_FACE_CONFIDENCE:
                return True
    return False
