# app.py
import os
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from pymongo import MongoClient
from werkzeug.utils import secure_filename

import base64
import cv2
import numpy as np
from datetime import datetime
from flask import jsonify


# import analyze_frame_from_array
from yolo_scripts.select_area_analysis import analyze_frame_from_array


# YOLO helpers (you'll create these in yolo_scripts/)
from yolo_scripts.crowd_image_upload_count import analyze_crowd
from yolo_scripts.crowd_video_analysis import analyze_video_full, analyze_video_single_frame
from yolo_scripts.select_area_analysis import analyze_selected_area

# ---------- CONFIG ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(STATIC_DIR, "results")
VIDEOS_DIR = os.path.join(STATIC_DIR, "videos")
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

# create required dirs
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "your_secret_key"  # change in prod

# ---------- MongoDB ----------
client = MongoClient("mongodb+srv://sheteyash001_db_user:Yash%40024@crowd-analytics.nonuzo7.mongodb.net/")
db = client["crowd_analytics"]
users_collection = db["users"]

# ---------- HELPERS ----------
def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("login.html")

# LOGIN
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")
    user = users_collection.find_one({"email": email})
    if user and user["password"] == password:
        session["user"] = email
        return redirect(url_for("dashboard"))
    else:
        flash("Invalid email or password")
        return redirect(url_for("home"))

# REGISTER
@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        flash("Invalid email format")
        return redirect(url_for("register_page"))

    if users_collection.find_one({"email": email}):
        flash("Email already exists")
        return redirect(url_for("register_page"))

    users_collection.insert_one({"name": name, "email": email, "password": password})
    flash("Registration successful")
    return redirect(url_for("home"))

# DASHBOARD
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("dashboard.html")

# LIVE CAMERAS page (keeps previous name)
@app.route("/live_cameras")
def Live_Cameras():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("live_cameras.html")

# IMAGE ANALYSIS (uses your existing analyze_crowd)
@app.route("/image_analysis")
def image_analysis():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("image_analysis.html")

@app.route("/image_analysis", methods=["POST"])
def image_upload():
    if "user" not in session:
        return redirect(url_for("home"))

    if "image" not in request.files:
        flash("No image uploaded")
        return redirect(url_for("image_analysis"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        flash("Invalid image file")
        return redirect(url_for("image_analysis"))

    filename = secure_filename(file.filename)
    upload_path = os.path.join(STATIC_DIR, filename)
    file.save(upload_path)

    result_filename = "result_" + filename
    result_path = os.path.join(RESULTS_DIR, result_filename)

    saved_image, count = analyze_crowd(upload_path, result_path)

    return render_template("image_analysis.html", result_image=result_filename, people_count=count)

# ---------------- VIDEO ANALYSIS ----------------
@app.route("/video_analysis")
def video_analysis():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("video_analysis.html")

@app.route("/video_analysis", methods=["POST"])
def process_video():
    if "user" not in session:
        return redirect(url_for("home"))

    if "video" not in request.files:
        flash("No video uploaded")
        return redirect(url_for("video_analysis"))

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        flash("Invalid video file")
        return redirect(url_for("video_analysis"))

    filename = secure_filename(file.filename)
    input_path = os.path.join(VIDEOS_DIR, filename)
    file.save(input_path)

    # method: "full" or "single"
    method = request.form.get("method", "full")

    if method == "single":
        # single frame extraction returns (result_image_filename, people_count)
        result_image_filename, people_count = analyze_video_single_frame(input_path, RESULTS_DIR)
        return render_template("video_analysis.html", result_image=result_image_filename, people_count=people_count, method=method)
    else:
        # full processing returns (result_video_filename, total_people)
        result_video_filename, total_people = analyze_video_full(input_path, RESULTS_DIR, VIDEOS_DIR)
        return render_template("video_analysis.html", result_video=result_video_filename, people_count=total_people, method=method)

# ---------------- AREA (ROI) ANALYSIS ----------------
# === Select Area from Video page (interactive) ===
@app.route("/select_area_video")
def select_area_video():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("select_area_video.html")

# Endpoint to accept a captured frame (base64) and ROI coords, analyze ROI, return JSON
@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    if "user" not in session:
        return jsonify({"error": "not authenticated"}), 403

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no data"}), 400

    data_url = data.get("image")
    x1 = data.get("x1")
    y1 = data.get("y1")
    x2 = data.get("x2")
    y2 = data.get("y2")

    if not data_url:
        return jsonify({"error": "no image data"}), 400

    try:
        # parse base64 dataURL
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

        # ensure coords are ints
        x1 = int(round(float(x1)))
        y1 = int(round(float(y1)))
        x2 = int(round(float(x2)))
        y2 = int(round(float(y2)))
    except Exception as e:
        return jsonify({"error": f"invalid payload: {str(e)}"}), 400

    # save annotated result with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"area_video_frame_{ts}.jpg"
    RESULTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "static", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, out_name)

    try:
        people_count = analyze_frame_from_array(frame, out_path, (x1, y1, x2, y2))
    except Exception as e:
        return jsonify({"error": f"analysis failed: {str(e)}"}), 500

    return jsonify({
        "people_count": int(people_count),
        "result_image": out_name
    })

# LOGOUT
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# Serve result files (optional)
@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
