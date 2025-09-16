import os
import cv2
import torch
import zipfile
from datetime import datetime
from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS
import numpy as np
import insightface

# ------------------- Flask setup -------------------
app = Flask(__name__)
CORS(app)

os.makedirs("uploads", exist_ok=True)       # known faces
os.makedirs("videos", exist_ok=True)        # uploaded videos
os.makedirs("reports", exist_ok=True)       # photo reports
os.makedirs("text_reports", exist_ok=True)  # plain text reports

# ------------------- Device & Model -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Using device: {device}")

# Load InsightFace ArcFace model
model = insightface.app.FaceAnalysis(name="buffalo_l")  # ArcFace-based
model.prepare(ctx_id=0 if device == "cuda" else -1)

known_faces = {}  # {name: embedding}

# ------------------- Load known faces -------------------
def load_known_faces():
    global known_faces
    print("[DEBUG] Loading known faces...")
    for person_name in os.listdir("uploads"):
        person_dir = os.path.join("uploads", person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = model.get(img)
            if not faces:
                continue
            emb = faces[0].normed_embedding
            known_faces[person_name] = emb
            print(f"[DEBUG] Added known face: {person_name}")
            break
    print(f"[DEBUG] Loaded known faces: {list(known_faces.keys())}")

load_known_faces()

# ------------------- Helper -------------------
def recognize_face(face_emb, threshold=0.8):
    """
    Compare ArcFace embeddings (cosine similarity).
    Higher similarity means closer match.
    """
    best_name = "Unknown"
    best_sim = threshold
    for name, ref_emb in known_faces.items():
        sim = np.dot(face_emb, ref_emb)  # embeddings are L2-normalized
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name

# ------------------- Video processing -------------------
def generate_frames(video_path, interval_seconds=2.5, max_width=640):
    cap = cv2.VideoCapture(video_path)
    print(f"[DEBUG] Streaming video: {video_path}")

    video_base = os.path.splitext(os.path.basename(video_path))[0]
    report_folder = os.path.join("reports", video_base)
    os.makedirs(report_folder, exist_ok=True)

    text_report_path = os.path.join("text_reports", f"{video_base}.txt")
    with open(text_report_path, "w") as f:
        f.write("Name | Timestamp\n")

    last_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time - last_time < interval_seconds:
            continue
        last_time = current_time
        frame_count += 1

        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Detect & recognize
        faces = model.get(frame)
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            name = recognize_face(f.normed_embedding)

            # Draw box + name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if name != "Unknown":
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Save cropped image
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    img_name = f"{name}_{frame_count}_{timestamp.replace(':','-').replace(' ','_')}.jpg"
                    cv2.imwrite(os.path.join(report_folder, img_name), face_img)

                # Append to text log
                with open(text_report_path, "a") as ftxt:
                    ftxt.write(f"{name} | {timestamp}\n")

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()
    print(f"[DEBUG] Finished streaming {video_path}")

# ------------------- Routes -------------------
@app.route("/upload", methods=["POST"])
def upload_video():
    file = request.files["video"]
    filepath = os.path.join("videos", file.filename)
    file.save(filepath)
    return jsonify({"video_name": file.filename})

@app.route("/stream/<video_name>")
def stream_video(video_name):
    path = os.path.join("videos", video_name)
    if not os.path.exists(path):
        return "Video not found", 404
    return Response(generate_frames(path),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/download_report/<video_name>")
def download_report(video_name):
    folder = os.path.join("reports", os.path.splitext(video_name)[0])
    if not os.path.exists(folder):
        return "Report not found", 404

    zip_path = f"{folder}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(folder):
            for f in files:
                zipf.write(os.path.join(root, f),
                           arcname=os.path.join(os.path.basename(folder), f))
    return send_file(zip_path, as_attachment=True)

@app.route("/download_text_report/<video_name>")
def download_text_report(video_name):
    txt = os.path.join("text_reports", f"{os.path.splitext(video_name)[0]}.txt")
    if not os.path.exists(txt):
        return "Text report not found", 404
    return send_file(txt, as_attachment=True)

# ------------------- Run -------------------
if __name__ == "__main__":
    print("====================================")
    print("🚀 ArcFace Face Recognition API running!")
    print("📡 http://localhost:5000")
    print("====================================")
    app.run(host="0.0.0.0", port=5000, debug=True)
