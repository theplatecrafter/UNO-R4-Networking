import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

import cv2
import numpy as np
import socket
import threading
import time
import json
import logging
import urllib.request
from collections import deque
from flask import Flask, render_template, Response, request, jsonify
from trackers.hsv_tracker import HSVTracker
# from trackers.yolo_tracker import YoloTracker

# Tracker System Initialization
active_trackers = {
    "hsv": HSVTracker(),
}
current_tracker = active_trackers["hsv"]
tracker_lock = threading.Lock()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- SYSTEM CONFIGURATION ---
CONFIG = {
    "phone_url": "",
    "arduino_ip": "",
    "arduino_port": 8888,
    "udp_listen_port": 8889,
    "camera_fov_deg": 60.0,
    "left_motor_in1": 2,
    "left_motor_in2": 3,
    "left_motor_en": 6,
    "right_motor_in1": 4,
    "right_motor_in2": 5,
    "right_motor_en": 9,
    "frame_rotation_deg": 0,
    "invert_left_motor": False,
    "invert_right_motor": False,
    "ultrasonic_trig_pin": 10,
    "ultrasonic_echo_pin": 11,
    "obstacle_stop_distance_cm": 20.0,
    "motor_speed_percent": 100,
    "tracking_duration_percent": 100,
    "manual_override": False,
    "manual_command": "STOP",
    "manual_last_seen": 0.0,
    "manual_queue_tail": 0.0,
    "tracking_enabled": True,
    "is_running": False
}

config_lock = threading.Lock()
reconnect_event = threading.Event()

udp_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

lower_hsv = np.array([0, 120, 70])
upper_hsv = np.array([10, 255, 255])
hsv_lock = threading.Lock()

telemetry = {
    "status": "Awaiting Setup...",
    "camera_connected": False,
    "arduino_status": "Not configured",
    "worker_alive": False,
    "last_error": "",
    "frames_received": 0,
    "target_angle": 0.0,
    "pipeline_latency_ms": 0,
    "fps": 0,
    "sampled_hsv": [0, 0, 0],
    "frame_rotation_deg": 0,
    "arduino_last_command": "None",
    "arduino_command_meaning": "No command sent",
    "arduino_commands_sent": 0,
    "arduino_last_sent_at": "Never"
    ,"motor_inversion": {"left": False, "right": False}
        ,"manual_override": False
        ,"manual_command": "STOP"
        ,"manual_queue_depth": 0
        ,"obstacle_distance_cm": -1.0
        ,"calibration_status": "Not calibrated"
        ,"tracking_enabled": True
        ,"motor_speed_percent": 100
        ,"tracking_duration_percent": 100
}

runtime_logs = deque(maxlen=200)
runtime_log_lock = threading.Lock()

def add_log(message, level="INFO"):
    entry = {"time": time.strftime("%H:%M:%S"), "level": level, "message": message}
    with runtime_log_lock:
        runtime_logs.append(entry)
    getattr(logging.getLogger("robot_tracker"), level.lower(), logging.info)(message)

latest_raw_frame = None
latest_processed_frame = None
frame_lock = threading.Lock()

# --- UDP TIME SYNCHRONIZATION SERVER ---
def udp_time_sync_listener():
    sync_sock = None
    bound_port = None
    while True:
        try:
            with config_lock:
                listen_port = CONFIG["udp_listen_port"]
            if sync_sock is None or listen_port != bound_port:
                if sync_sock is not None:
                    sync_sock.close()
                sync_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sync_sock.settimeout(1.0)
                sync_sock.bind(('0.0.0.0', listen_port))
                bound_port = listen_port
                add_log(f"Time sync listener bound to UDP port {bound_port}")
            data, addr = sync_sock.recvfrom(1024)
            message = data.decode().strip()
            if message == "SYNC_REQ":
                now = time.time()
                sync_sock.sendto(f"SYNC_ACK,{now:.4f}".encode(), addr)
            elif message.startswith("ULTRA_STATUS,"):
                try:
                    telemetry["obstacle_distance_cm"] = float(message.split(",", 1)[1])
                except ValueError:
                    pass
            elif message.startswith("CAL_STATUS,"):
                telemetry["calibration_status"] = message.split(",", 1)[1]
        except socket.timeout:
            continue
        except OSError as exc:
            add_log(f"Time sync listener error: {exc}", "ERROR")
            if sync_sock is not None:
                sync_sock.close()
            sync_sock = None
            time.sleep(1)
        except Exception:
            pass

threading.Thread(target=udp_time_sync_listener, daemon=True).start()

# --- VISION PROCESSING ENGINE ---
def process_video():
    global latest_raw_frame, latest_processed_frame, telemetry
    cap = None
    prev_time = time.time()
    fps_smooth = 0.0
    failed_reads = 0
    telemetry["worker_alive"] = True
    add_log("Vision worker started")

    while True:
        try:
            with config_lock:
                running = CONFIG["is_running"]
                cam_url = CONFIG["phone_url"]

            if not running:
                if cap is not None:
                    cap.release()
                    cap = None
                telemetry["camera_connected"] = False
                telemetry["status"] = "Engine Stopped"
                time.sleep(0.1)
                continue

            if not cam_url:
                telemetry["camera_connected"] = False
                telemetry["status"] = "Camera URL missing"
                time.sleep(0.2)
                continue

            if cap is None or reconnect_event.is_set():
                reconnect_event.clear()
                if cap is not None:
                    cap.release()
                add_log(f"Connecting to camera: {cam_url}")
                cap = cv2.VideoCapture(cam_url)
                if not cap.isOpened():
                    telemetry["camera_connected"] = False
                    telemetry["status"] = "Camera connection failed"
                    telemetry["last_error"] = f"Could not open {cam_url}"
                    add_log(telemetry["last_error"], "ERROR")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue
                telemetry["camera_connected"] = True
                telemetry["last_error"] = ""
                add_log("Camera stream opened")

            capture_time = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                telemetry["camera_connected"] = False
                telemetry["status"] = "Camera read failed"
                telemetry["last_error"] = f"No frame received (attempt {failed_reads})"
                if failed_reads == 1 or failed_reads % 30 == 0:
                    add_log(telemetry["last_error"], "WARN")
                if failed_reads >= 10:
                    cap.release()
                    cap = None
                    failed_reads = 0
                time.sleep(0.05)
                continue
            failed_reads = 0
            telemetry["camera_connected"] = True
            telemetry["frames_received"] += 1

            with config_lock:
                rotation = CONFIG["frame_rotation_deg"]
            telemetry["frame_rotation_deg"] = rotation
            raw_frame = frame
            if rotation == 90:
                raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_180)
            elif rotation == 270:
                raw_frame = cv2.rotate(raw_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            raw_frame = cv2.resize(raw_frame, (640, 480))
            height, width, _ = raw_frame.shape
            center_x = width // 2

            # --- MODULAR TRACKER INJECTION ---
            with config_lock:
                manual_override = CONFIG["manual_override"]
                manual_command = CONFIG["manual_command"]
                manual_stale = time.time() - CONFIG["manual_last_seen"] > 0.6
                tracking_enabled = CONFIG["tracking_enabled"]
            telemetry["tracking_enabled"] = tracking_enabled
            if manual_override:
                if manual_stale:
                    manual_command = "STOP"
                processed_frame = raw_frame.copy()
                payload = None
                telemetry["status"] = f"Remote control: {manual_command}"
                telemetry["manual_override"] = True
                telemetry["manual_command"] = manual_command
            elif not tracking_enabled:
                processed_frame = raw_frame.copy()
                payload = "NO_TARGET"
                telemetry["status"] = "Tracking disabled"
                telemetry["tracking_enabled"] = False
            else:
                with tracker_lock:
                    processed_frame, target_found, x, y, radius = current_tracker.process_frame(raw_frame)

                if target_found:
                    offset_x = x - center_x
                    angle_deg = (offset_x / center_x) * (CONFIG["camera_fov_deg"] / 2.0)
                    cv2.line(processed_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

                    payload = f"TARGET,{capture_time:.4f},{angle_deg:.2f},{radius:.1f},0.100"
                    telemetry["status"] = "Tracking"
                    telemetry["target_angle"] = round(angle_deg, 1)
                else:
                    payload = f"NO_TARGET,{capture_time:.4f}"
                    telemetry["status"] = "Searching"
                telemetry["manual_override"] = False
                telemetry["tracking_enabled"] = True

            if manual_override:
                command_meaning = f"Manual override: {manual_command}"
            elif not tracking_enabled:
                command_meaning = "Tracking disabled; no tracking command generated"
            elif target_found:
                command_meaning = (
                    f"Track target at {telemetry['target_angle']} degrees; "
                    f"estimated radius {radius:.1f}px"
                )
            else:
                command_meaning = "No target detected; stop motors"
            telemetry["arduino_last_command"] = payload or "Timed manual queue"
            telemetry["arduino_command_meaning"] = command_meaning

            # Tracking and safety commands are sent per frame; timed manual commands
            # are sent once by the manual-control endpoint.
            if payload is not None:
                with config_lock:
                    try:
                        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
                        udp_out.sendto(payload.encode(), arduino_address)
                        telemetry["arduino_status"] = f"UDP packets sent to {arduino_address[0]}:{arduino_address[1]}"
                        telemetry["arduino_commands_sent"] += 1
                        telemetry["arduino_last_sent_at"] = time.strftime("%H:%M:%S")
                    except Exception as exc:
                        telemetry["arduino_status"] = "UDP send failed"
                        telemetry["last_error"] = str(exc)
                        add_log(f"Arduino UDP send failed: {exc}", "ERROR")

            telemetry["pipeline_latency_ms"] = round((time.time() - capture_time) * 1000.0, 1)
        
            # FPS calc & frame storage (same as before)
            curr_time = time.time()
            fps_smooth = (0.1 * (1.0 / (curr_time - prev_time))) + (0.9 * fps_smooth) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            telemetry["fps"] = round(fps_smooth, 1)

            with frame_lock:
                latest_raw_frame = raw_frame.copy()
                latest_processed_frame = processed_frame.copy()

        except Exception as exc:
            telemetry["camera_connected"] = False
            telemetry["status"] = "Vision worker error"
            telemetry["last_error"] = str(exc)
            add_log(f"Vision worker error: {exc}", "ERROR")
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.5)


threading.Thread(target=process_video, name="vision-worker", daemon=True).start()
            
            
# --- HELPER: AUTO-DETECT LAPTOP IP ---
def get_local_ip():
    """Tricks the OS into revealing the primary local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # We don't actually need to connect to this IP, 
        # it just forces the OS to evaluate which network interface to use.
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '192.168.43.1' # Fallback
    finally:
        s.close()
    return IP

            
# --- FLASK ROUTES ---
def generate_frames(stream_type="processed"):
    while True:
        with frame_lock:
            target_frame = latest_raw_frame if stream_type == "raw" else latest_processed_frame
            if target_frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', target_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index(): return render_template('index.html', auto_ip=get_local_ip())

@app.route('/video_feed/raw')
def video_feed_raw(): return Response(generate_frames("raw"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/processed')
def video_feed_processed(): return Response(generate_frames("processed"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_color', methods=['POST'])
def select_color():
    data = request.get_json()
    with frame_lock:
        if latest_raw_frame is None: return jsonify({"status": "error"})
        pixel = cv2.cvtColor(latest_raw_frame, cv2.COLOR_BGR2HSV)[min(data.get('y',0), 479), min(data.get('x',0), 639)]
    
    h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])
    
    # Pass dynamic updates back to the active modular tracker!
    with tracker_lock:
        current_tracker.update_params({
            "lower": [max(0, h - 15), max(40, s - 70), max(40, v - 70)],
            "upper": [min(180, h + 15), min(255, s + 70), min(255, v + 70)]
        })
    
    telemetry["sampled_hsv"] = [h, s, v]
    return jsonify({"status": "success"})

@app.route('/api/set_tracker', methods=['POST'])
def set_tracker():
    """Allows UI to swap tracking models on the fly"""
    global current_tracker
    model_name = request.get_json().get("model")
    if model_name in active_trackers:
        with tracker_lock:
            current_tracker = active_trackers[model_name]
        return jsonify({"status": "success", "active": model_name})
    return jsonify({"status": "error"}), 400

@app.route('/telemetry_stream')
def telemetry_stream():
    def stream():
        while True:
            yield f"data: {json.dumps(telemetry)}\n\n"
            time.sleep(0.05)
    return Response(stream(), mimetype='text/event-stream')

@app.route('/api/logs')
def get_logs():
    with runtime_log_lock:
        return jsonify(list(runtime_logs))

# --- WIZARD API ENDPOINTS ---
@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        data = request.get_json()
        try:
            phone_ip = str(data.get("phone_ip", "")).strip()
            phone_port = int(data.get("phone_port", 8080))
            arduino_ip = str(data.get("arduino_ip", "")).strip()
            arduino_port = int(data.get("arduino_port", 8888))
            udp_listen_port = int(data.get("udp_listen_port", CONFIG["udp_listen_port"]))
            camera_fov_deg = float(data.get("camera_fov_deg", CONFIG["camera_fov_deg"]))
            motor_pins = {
                key: int(data.get(key, CONFIG[key]))
                for key in ("left_motor_in1", "left_motor_in2", "left_motor_en", "right_motor_in1", "right_motor_in2", "right_motor_en")
            }
            frame_rotation_deg = int(data.get("frame_rotation_deg", CONFIG["frame_rotation_deg"]))
            invert_left_motor = bool(data.get("invert_left_motor", CONFIG["invert_left_motor"]))
            invert_right_motor = bool(data.get("invert_right_motor", CONFIG["invert_right_motor"]))
            ultrasonic_trig_pin = int(data.get("ultrasonic_trig_pin", CONFIG["ultrasonic_trig_pin"]))
            ultrasonic_echo_pin = int(data.get("ultrasonic_echo_pin", CONFIG["ultrasonic_echo_pin"]))
            obstacle_stop_distance_cm = float(data.get("obstacle_stop_distance_cm", CONFIG["obstacle_stop_distance_cm"]))
            motor_speed_percent = int(data.get("motor_speed_percent", CONFIG["motor_speed_percent"]))
            tracking_duration_percent = int(data.get("tracking_duration_percent", CONFIG["tracking_duration_percent"]))
            if not phone_ip or not arduino_ip or not 1 <= phone_port <= 65535 or not 1 <= arduino_port <= 65535:
                raise ValueError("Camera and Arduino IP addresses and valid ports are required")
            if any(not 0 <= pin <= 53 for pin in motor_pins.values()):
                raise ValueError("Motor pins must be Arduino pin numbers from 0 to 53")
            if len(set(motor_pins.values())) != len(motor_pins):
                raise ValueError("Each motor pin must be unique")
            if frame_rotation_deg not in (0, 90, 180, 270):
                raise ValueError("Frame rotation must be 0, 90, 180, or 270 degrees")
            if not 0 <= ultrasonic_trig_pin <= 53 or not 0 <= ultrasonic_echo_pin <= 53:
                raise ValueError("Ultrasonic pins must be Arduino pin numbers from 0 to 53")
            if ultrasonic_trig_pin == ultrasonic_echo_pin:
                raise ValueError("Ultrasonic Trig and Echo pins must be different")
            if obstacle_stop_distance_cm <= 0:
                raise ValueError("Obstacle stop distance must be greater than zero")
            if not 0 <= motor_speed_percent <= 100:
                raise ValueError("Motor speed must be between 0 and 100 percent")
            if not 0 <= tracking_duration_percent <= 200:
                raise ValueError("Tracking instruction duration must be between 0 and 200 percent")
            with config_lock:
                CONFIG["phone_url"] = f"http://{phone_ip}:{phone_port}/video"
                CONFIG["arduino_ip"] = arduino_ip
                CONFIG["arduino_port"] = arduino_port
                CONFIG["udp_listen_port"] = udp_listen_port
                CONFIG["camera_fov_deg"] = camera_fov_deg
                CONFIG.update(motor_pins)
                CONFIG["frame_rotation_deg"] = frame_rotation_deg
                CONFIG["invert_left_motor"] = invert_left_motor
                CONFIG["invert_right_motor"] = invert_right_motor
                CONFIG["ultrasonic_trig_pin"] = ultrasonic_trig_pin
                CONFIG["ultrasonic_echo_pin"] = ultrasonic_echo_pin
                CONFIG["obstacle_stop_distance_cm"] = obstacle_stop_distance_cm
                CONFIG["motor_speed_percent"] = motor_speed_percent
                CONFIG["tracking_duration_percent"] = tracking_duration_percent
                telemetry["arduino_status"] = f"Configured for {arduino_ip}:{arduino_port}"
                telemetry["motor_inversion"] = {"left": invert_left_motor, "right": invert_right_motor}
                telemetry["motor_speed_percent"] = motor_speed_percent
                telemetry["tracking_duration_percent"] = tracking_duration_percent
        except (TypeError, ValueError) as exc:
            add_log(f"Configuration rejected: {exc}", "ERROR")
            return jsonify({"status": "error", "message": str(exc)}), 400
        add_log(f"Configuration saved. Camera: {CONFIG['phone_url']}; Arduino: {CONFIG['arduino_ip']}:{CONFIG['arduino_port']}")
        inversion_packet = f"INVERT,{int(invert_left_motor)},{int(invert_right_motor)}"
        try:
            udp_out.sendto(inversion_packet.encode(), (arduino_ip, arduino_port))
            add_log(f"Sent motor inversion to Arduino: {inversion_packet}")
            speed_packet = f"SPEED,{motor_speed_percent}"
            udp_out.sendto(speed_packet.encode(), (arduino_ip, arduino_port))
            add_log(f"Sent motor speed to Arduino: {speed_packet}")
            duration_packet = f"TRACK_DURATION,{tracking_duration_percent}"
            udp_out.sendto(duration_packet.encode(), (arduino_ip, arduino_port))
            add_log(f"Sent tracking duration to Arduino: {duration_packet}")
        except OSError as exc:
            add_log(f"Could not send initial motor inversion: {exc}", "WARN")
        reconnect_event.set()
        return jsonify({"status": "success"})
    with config_lock:
        config = dict(CONFIG)
    config["phone_ip"] = config["phone_url"].removeprefix("http://").rsplit(":", 1)[0] if config["phone_url"] else ""
    config["phone_port"] = int(config["phone_url"].rsplit(":", 1)[1].split("/", 1)[0]) if config["phone_url"] else 8080
    return jsonify(config)

@app.route('/api/control', methods=['POST'])
def handle_control():
    action = request.get_json().get("action")
    with config_lock:
        CONFIG["is_running"] = (action == "start")
    if action == "start":
        reconnect_event.set()
        with config_lock:
            arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
        try:
            udp_out.sendto(b"CALIBRATE", arduino_address)
            telemetry["calibration_status"] = "Calibration requested"
            add_log("Engine start requested motor calibration")
        except OSError as exc:
            add_log(f"Could not request motor calibration: {exc}", "ERROR")
    telemetry["status"] = "Starting engine..." if action == "start" else "Engine Stopped"
    add_log(f"Engine command: {action}")
    return jsonify({"status": "success"})

@app.route('/api/calibrate', methods=['POST'])
def calibrate_motors():
    with config_lock:
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    try:
        udp_out.sendto(b"CALIBRATE", arduino_address)
        telemetry["calibration_status"] = "Calibration requested"
        add_log("Motor calibration requested")
        return jsonify({"status": "success"})
    except OSError as exc:
        add_log(f"Could not request motor calibration: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/motor_inversion', methods=['POST'])
def set_motor_inversion():
    data = request.get_json(silent=True) or {}
    with config_lock:
        if "left" in data:
            CONFIG["invert_left_motor"] = bool(data["left"])
        if "right" in data:
            CONFIG["invert_right_motor"] = bool(data["right"])
        left_inverted = CONFIG["invert_left_motor"]
        right_inverted = CONFIG["invert_right_motor"]
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    telemetry["motor_inversion"] = {"left": left_inverted, "right": right_inverted}
    packet = f"INVERT,{int(left_inverted)},{int(right_inverted)}"
    try:
        udp_out.sendto(packet.encode(), arduino_address)
        add_log(f"Motor inversion updated: left={left_inverted}, right={right_inverted}")
        return jsonify({"status": "success", "packet": packet})
    except OSError as exc:
        add_log(f"Could not send motor inversion to Arduino: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/motor_speed', methods=['POST'])
def set_motor_speed():
    data = request.get_json(silent=True) or {}
    try:
        speed_percent = int(data.get("percent"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Motor speed must be an integer from 0 to 100"}), 400
    if not 0 <= speed_percent <= 100:
        return jsonify({"status": "error", "message": "Motor speed must be between 0 and 100 percent"}), 400
    with config_lock:
        CONFIG["motor_speed_percent"] = speed_percent
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    telemetry["motor_speed_percent"] = speed_percent
    packet = f"SPEED,{speed_percent}"
    try:
        udp_out.sendto(packet.encode(), arduino_address)
        add_log(f"Motor speed set to {speed_percent}%")
        return jsonify({"status": "success", "percent": speed_percent, "packet": packet})
    except OSError as exc:
        add_log(f"Could not update motor speed: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/tracking_duration', methods=['POST'])
def set_tracking_duration():
    data = request.get_json(silent=True) or {}
    try:
        duration_percent = int(data.get("percent"))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Tracking duration must be an integer from 0 to 200"}), 400
    if not 0 <= duration_percent <= 200:
        return jsonify({"status": "error", "message": "Tracking duration must be between 0 and 200 percent"}), 400
    with config_lock:
        CONFIG["tracking_duration_percent"] = duration_percent
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    telemetry["tracking_duration_percent"] = duration_percent
    packet = f"TRACK_DURATION,{duration_percent}"
    try:
        udp_out.sendto(packet.encode(), arduino_address)
        add_log(f"Tracking instruction duration set to {duration_percent}%")
        return jsonify({"status": "success", "percent": duration_percent, "packet": packet})
    except OSError as exc:
        add_log(f"Could not update tracking duration: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/manual_mode', methods=['POST'])
def set_manual_mode():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled", False))
    with config_lock:
        CONFIG["manual_override"] = enabled
        CONFIG["manual_command"] = "STOP"
        CONFIG["manual_last_seen"] = time.time()
        CONFIG["manual_queue_tail"] = time.time()
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    telemetry["manual_override"] = enabled
    telemetry["manual_command"] = "STOP"
    telemetry["manual_queue_depth"] = 0
    try:
        udp_out.sendto(b"MANUAL,STOP", arduino_address)
        add_log(f"Manual override {'enabled' if enabled else 'disabled'}; motors stopped")
        return jsonify({"status": "success", "enabled": enabled})
    except OSError as exc:
        add_log(f"Could not update manual mode: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/tracking_mode', methods=['POST'])
def set_tracking_mode():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled", True))
    with config_lock:
        CONFIG["tracking_enabled"] = enabled
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    telemetry["tracking_enabled"] = enabled
    if not enabled:
        try:
            udp_out.sendto(b"NO_TARGET", arduino_address)
        except OSError as exc:
            add_log(f"Could not stop tracking output: {exc}", "ERROR")
            return jsonify({"status": "error", "message": str(exc)}), 500
    add_log(f"Tracking calculations {'enabled' if enabled else 'disabled'}")
    return jsonify({"status": "success", "enabled": enabled})

@app.route('/api/manual_control', methods=['POST'])
def manual_control():
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "STOP")).upper()
    if command not in {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}:
        return jsonify({"status": "error", "message": "Unknown manual command"}), 400
    duration = float(data.get("duration_seconds", 0))
    if duration < 0 or duration > 60:
        return jsonify({"status": "error", "message": "Duration must be between 0 and 60 seconds"}), 400
    with config_lock:
        if not CONFIG["manual_override"]:
            return jsonify({"status": "error", "message": "Manual override is disabled"}), 409
        CONFIG["manual_command"] = command
        CONFIG["manual_last_seen"] = time.time()
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
        if duration > 0 and command != "STOP":
            start_time = max(time.time(), CONFIG["manual_queue_tail"])
            CONFIG["manual_queue_tail"] = start_time + duration
        else:
            start_time = time.time()
            CONFIG["manual_queue_tail"] = start_time
    telemetry["manual_command"] = command
    try:
        if duration > 0 and command != "STOP":
            packet = f"MANUAL_TIMED,{command},{start_time:.4f},{duration:.4f}"
            telemetry["manual_queue_depth"] += 1
        else:
            packet = "MANUAL,STOP"
            telemetry["manual_queue_depth"] = 0
        udp_out.sendto(packet.encode(), arduino_address)
        add_log(f"Queued manual command: {packet}")
        return jsonify({"status": "success", "command": command, "packet": packet})
    except OSError as exc:
        add_log(f"Could not send manual command: {exc}", "ERROR")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route('/api/rotate_frame', methods=['POST'])
def rotate_frame():
    requested_rotation = request.get_json(silent=True) or {}
    with config_lock:
        current_rotation = CONFIG["frame_rotation_deg"]
        rotation = requested_rotation.get("degrees", (current_rotation + 90) % 360)
        try:
            rotation = int(rotation)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Rotation must be 0, 90, 180, or 270 degrees"}), 400
        if rotation not in (0, 90, 180, 270):
            return jsonify({"status": "error", "message": "Rotation must be 0, 90, 180, or 270 degrees"}), 400
        CONFIG["frame_rotation_deg"] = rotation
    telemetry["frame_rotation_deg"] = rotation
    add_log(f"Camera frame rotation set to {rotation} degrees")
    return jsonify({"status": "success", "frame_rotation_deg": rotation})

@app.route('/api/test_camera', methods=['POST'])
def test_camera():
    data = request.get_json()
    url = f"http://{data.get('ip')}:{data.get('port')}/video"
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                add_log(f"Camera test succeeded: {url}")
                return jsonify({"status": "success"})
    except Exception as e:
        add_log(f"Camera test failed for {url}: {e}", "ERROR")
        return jsonify({"status": "error", "message": str(e)}), 400
    return jsonify({"status": "error", "message": "Camera returned an unexpected status"}), 400

@app.route('/api/download_ino', methods=['GET'])
def download_ino():
    ssid = request.args.get('ssid', '')
    password = request.args.get('password', '')
    laptop_ip = request.args.get('laptop_ip', '192.168.43.1')
    invert_left_motor = request.args.get('invert_left_motor', '0') == '1'
    invert_right_motor = request.args.get('invert_right_motor', '0') == '1'
    ultrasonic_trig_pin = int(request.args.get('ultrasonic_trig_pin', 10))
    ultrasonic_echo_pin = int(request.args.get('ultrasonic_echo_pin', 11))
    obstacle_stop_distance_cm = float(request.args.get('obstacle_stop_distance_cm', 20.0))
    motor_speed_percent = int(request.args.get('motor_speed_percent', 100))
    tracking_duration_percent = int(request.args.get('tracking_duration_percent', 100))
    if not 0 <= ultrasonic_trig_pin <= 53 or not 0 <= ultrasonic_echo_pin <= 53 or ultrasonic_trig_pin == ultrasonic_echo_pin or obstacle_stop_distance_cm <= 0 or not 0 <= motor_speed_percent <= 100 or not 0 <= tracking_duration_percent <= 200:
        return jsonify({"status": "error", "message": "Invalid HC-SR04 pins or stop distance"}), 400
    motor_pin_defaults = {
        "left_motor_in1": 2, "left_motor_in2": 3, "left_motor_en": 6,
        "right_motor_in1": 4, "right_motor_in2": 5, "right_motor_en": 9
    }
    try:
        motor_pins = {key: int(request.args.get(key, default)) for key, default in motor_pin_defaults.items()}
        if any(not 0 <= pin <= 53 for pin in motor_pins.values()) or len(set(motor_pins.values())) != len(motor_pins):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Motor pins must be unique Arduino pin numbers from 0 to 53"}), 400

    # FULL Arduino Code with Latency Compensation injected via string replace (safest method for C++ in Python)
    ino_code = """#include <WiFiS3.h>
#include <WiFiUdp.h>

const char ssid[] = "__SSID__";
const char pass[] = "__PASS__";

// Time Sync Config
IPAddress laptopIP(__LAPTOP_IP_COMMAS__); 
const unsigned int localPort = 8888;
const unsigned int laptopSyncPort = 8889;

WiFiUDP Udp;
char packetBuffer[255];

double laptopTimeOffsetSec = 0.0;
const float ROBOT_TURN_SPEED_DPS = 45.0;
float currentAngularVelocity = 0.0;
bool invertLeftMotor = __INVERT_LEFT_MOTOR__;
bool invertRightMotor = __INVERT_RIGHT_MOTOR__;
const int ULTRASONIC_TRIG = __ULTRASONIC_TRIG_PIN__;
const int ULTRASONIC_ECHO = __ULTRASONIC_ECHO_PIN__;
const float OBSTACLE_STOP_DISTANCE_CM = __OBSTACLE_STOP_DISTANCE_CM__;
float calibratedForwardSpeedCmS = 0.0;
int motorSpeedPercent = __MOTOR_SPEED_PERCENT__;
int trackingDurationPercent = __TRACKING_DURATION_PERCENT__;
double trackingCommandUntil = 0.0;
struct TimedCommand {
    String command;
    double startTime;
    double duration;
};
TimedCommand commandQueue[16];
int queueHead = 0;
int queueTail = 0;
bool timedCommandActive = false;
int activeTimedCommand = 0;
double activeTimedEnd = 0.0;
unsigned long lastObstacleCheck = 0;

void syncTimeWithLaptop();
double getSyncedTime();
void parseAndExecuteCommand(String packet);
void driveProportional(float angleError, float radius);
void driveManual(int command);
void enqueueTimedCommand(String command, double startTime, double duration);
int commandNumber(String command);
void runTimedCommands();
void setMotor(int in1, int in2, int enablePin, bool forward, int speed, bool inverted);
float readDistanceCm();
void sendStatus(String status);
bool obstacleTooClose();
void calibrateForwardSpeed();
void stopMotors();

// Motor Pins (L298N)
const int LEFT_EN = __LEFT_MOTOR_EN__; const int LEFT_IN1 = __LEFT_MOTOR_IN1__; const int LEFT_IN2 = __LEFT_MOTOR_IN2__;
const int RIGHT_IN1 = __RIGHT_MOTOR_IN1__; const int RIGHT_IN2 = __RIGHT_MOTOR_IN2__; const int RIGHT_EN = __RIGHT_MOTOR_EN__;

void setup() {
  Serial.begin(115200);
    Serial.println("[BOOT] Robot controller starting");
    Serial.print("[PINS] Left motor IN1/IN2/EN: "); Serial.print(LEFT_IN1); Serial.print("/"); Serial.print(LEFT_IN2); Serial.print("/"); Serial.println(LEFT_EN);
    Serial.print("[PINS] Right motor IN1/IN2/EN: "); Serial.print(RIGHT_IN1); Serial.print("/"); Serial.print(RIGHT_IN2); Serial.print("/"); Serial.println(RIGHT_EN);
    pinMode(LEFT_EN, OUTPUT); pinMode(RIGHT_EN, OUTPUT);
    pinMode(LEFT_IN1, OUTPUT); pinMode(LEFT_IN2, OUTPUT);
    pinMode(RIGHT_IN1, OUTPUT); pinMode(RIGHT_IN2, OUTPUT);
  stopMotors();
    pinMode(ULTRASONIC_TRIG, OUTPUT); pinMode(ULTRASONIC_ECHO, INPUT);

  WiFi.begin(ssid, pass);
    Serial.println("[WIFI] Connecting to configured hotspot...");
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print("."); 
  }
  
  // Force the Arduino to wait until the phone assigns an actual IP
  IPAddress ip = WiFi.localIP();
  while (ip[0] == 0) {
    delay(500);
    Serial.print("~"); 
    ip = WiFi.localIP();
  }

  Serial.println("\\nWiFi Connected! IP: "); 
  Serial.println(ip);
  
  Udp.begin(localPort);
    Serial.print("[UDP] Listening on port "); Serial.println(localPort);
  syncTimeWithLaptop();
}

void loop() {
  static unsigned long lastSyncCheck = 0;
  if (millis() - lastSyncCheck > 30000) { syncTimeWithLaptop(); lastSyncCheck = millis(); }

  int packetSize = Udp.parsePacket();
  if (packetSize) {
        Serial.print("[UDP] Packet received, bytes: "); Serial.println(packetSize);
    int len = Udp.read(packetBuffer, 254);
    if (len > 0) packetBuffer[len] = 0;
        Serial.print("[UDP] Raw command: "); Serial.println(packetBuffer);
    parseAndExecuteCommand(String(packetBuffer));
  }
    runTimedCommands();
    if (trackingCommandUntil > 0 && getSyncedTime() >= trackingCommandUntil) {
        stopMotors();
        trackingCommandUntil = 0.0;
    }
}

void syncTimeWithLaptop() {
  Udp.beginPacket(laptopIP, laptopSyncPort);
  Udp.write("SYNC_REQ");
  Udp.endPacket();

  unsigned long startWait = millis();
  while (millis() - startWait < 1000) {
    if (Udp.parsePacket()) {
    int len = Udp.read(packetBuffer, 254);
      packetBuffer[len] = 0;
      String msg = String(packetBuffer);
      if (msg.startsWith("SYNC_ACK")) {
        double laptopTime = msg.substring(9).toDouble();
        laptopTimeOffsetSec = laptopTime - (millis() / 1000.0);
                Serial.print("[SYNC] Clock offset seconds: "); Serial.println(laptopTimeOffsetSec, 4);
        return;
      }
    }
  }
    Serial.println("[SYNC] Laptop did not respond within 1 second");
}

double getSyncedTime() { return (millis() / 1000.0) + laptopTimeOffsetSec; }

void parseAndExecuteCommand(String packet) {
    if (packet.startsWith("TRACK_DURATION")) {
        trackingDurationPercent = constrain(packet.substring(packet.indexOf(',') + 1).toInt(), 0, 200);
        Serial.print("[TRACKING] Instruction duration set to "); Serial.print(trackingDurationPercent); Serial.println("%");
    } else if (packet.startsWith("SPEED")) {
        motorSpeedPercent = packet.substring(packet.indexOf(',') + 1).toInt();
        motorSpeedPercent = constrain(motorSpeedPercent, 0, 100);
        Serial.print("[MOTOR] Speed set to "); Serial.print(motorSpeedPercent); Serial.println("%");
    } else if (packet.startsWith("MANUAL_TIMED")) {
        int first = packet.indexOf(',');
        int second = packet.indexOf(',', first + 1);
        int third = packet.indexOf(',', second + 1);
        String command = packet.substring(first + 1, second);
        double startTime = packet.substring(second + 1, third).toDouble();
        double duration = packet.substring(third + 1).toDouble();
        enqueueTimedCommand(command, startTime, duration);
    } else if (packet.startsWith("CALIBRATE")) {
        calibrateForwardSpeed();
    } else if (packet.startsWith("MANUAL")) {
        String command = packet.substring(packet.indexOf(',') + 1);
        Serial.print("[MANUAL] Command: "); Serial.println(command);
        if (command == "FORWARD") driveManual(1);
        else if (command == "BACKWARD") driveManual(2);
        else if (command == "LEFT") driveManual(3);
        else if (command == "RIGHT") driveManual(4);
        else {
            queueHead = queueTail;
            timedCommandActive = false;
            driveManual(0);
        }
    } else if (packet.startsWith("INVERT")) {
        int comma = packet.indexOf(',');
        int secondComma = packet.indexOf(',', comma + 1);
        invertLeftMotor = packet.substring(comma + 1, secondComma).toInt() != 0;
        invertRightMotor = packet.substring(secondComma + 1).toInt() != 0;
        Serial.print("[MOTOR] Inversion updated, left/right: ");
        Serial.print(invertLeftMotor); Serial.print("/"); Serial.println(invertRightMotor);
    } else if (packet.startsWith("TARGET")) {
    int idx1 = packet.indexOf(',');
    int idx2 = packet.indexOf(',', idx1 + 1);
    int idx3 = packet.indexOf(',', idx2 + 1);
    int idx4 = packet.indexOf(',', idx3 + 1);

    double captureTime = packet.substring(idx1 + 1, idx2).toDouble();
    float rawAngleDeg = packet.substring(idx2 + 1, idx3).toFloat();
    float radius = idx4 > 0 ? packet.substring(idx3 + 1, idx4).toFloat() : packet.substring(idx3 + 1).toFloat();
    float baseDuration = idx4 > 0 ? packet.substring(idx4 + 1).toFloat() : 0.100;

    Serial.print("[TARGET] Capture time: "); Serial.println(captureTime, 4);
    Serial.print("[TARGET] Raw angle: "); Serial.print(rawAngleDeg, 2);
    Serial.print(" deg, radius: "); Serial.println(radius, 1);

    float latencySec = (float)(getSyncedTime() - captureTime);
    if (latencySec < 0 || latencySec > 2.0) latencySec = 0;

    // Latency Compensation Math
    float turnedDuringDelay = currentAngularVelocity * latencySec;
    float correctedAngleDeg = rawAngleDeg - turnedDuringDelay;

        Serial.print("[TARGET] Latency: "); Serial.print(latencySec, 3);
        Serial.print(" s, corrected angle: "); Serial.println(correctedAngleDeg, 2);

    driveProportional(correctedAngleDeg, radius);
    trackingCommandUntil = getSyncedTime() + (baseDuration * trackingDurationPercent / 100.0);
  } else if (packet.startsWith("NO_TARGET")) {
        Serial.println("[TARGET] No target detected; stopping motors");
    stopMotors(); currentAngularVelocity = 0.0;
    trackingCommandUntil = 0.0;
    } else {
        Serial.println("[UDP] Unknown command ignored");
  }
}

void driveProportional(float angleError, float radius) {
  int baseSpeed = 160;   
  float kp = 2.5;        

    if (radius > 120) {
        Serial.println("[MOTOR] Target too large/close; stopping");
        stopMotors(); currentAngularVelocity = 0.0; return;
    }

  int turnAdjust = (int)(abs(angleError) * kp);
  int leftSpeed = constrain(baseSpeed + turnAdjust, 0, 255);
  int rightSpeed = constrain(baseSpeed + turnAdjust, 0, 255);

  if (angleError < -5.0) { // Left
        Serial.print("[MOTOR] Turning left, PWM: "); Serial.println(leftSpeed);
    setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, false, leftSpeed, invertLeftMotor);
    setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, true, rightSpeed, invertRightMotor);
    currentAngularVelocity = -ROBOT_TURN_SPEED_DPS;
  } else if (angleError > 5.0) { // Right
        Serial.print("[MOTOR] Turning right, PWM: "); Serial.println(rightSpeed);
    setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, true, leftSpeed, invertLeftMotor);
    setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, false, rightSpeed, invertRightMotor);
    currentAngularVelocity = ROBOT_TURN_SPEED_DPS;
  } else { // Forward
        Serial.print("[MOTOR] Driving forward, PWM: "); Serial.println(baseSpeed);
    if (obstacleTooClose()) {
        Serial.println("[SAFETY] Obstacle within stop distance; stopping");
        stopMotors(); currentAngularVelocity = 0.0; return;
    }
    setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, true, baseSpeed, invertLeftMotor);
    setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, true, baseSpeed, invertRightMotor);
    currentAngularVelocity = 0.0;
  }
}

void driveManual(int command) {
    if (command == 0) {
        Serial.println("[MANUAL] Stop");
        stopMotors(); currentAngularVelocity = 0.0; return;
    }
    if (command == 1) {
        Serial.println("[MANUAL] Forward");
        if (obstacleTooClose()) {
            Serial.println("[SAFETY] Obstacle within stop distance; stopping");
            stopMotors(); currentAngularVelocity = 0.0; return;
        }
        setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, true, 160, invertLeftMotor);
        setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, true, 160, invertRightMotor);
    } else if (command == 2) {
        Serial.println("[MANUAL] Backward");
        setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, false, 160, invertLeftMotor);
        setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, false, 160, invertRightMotor);
    } else if (command == 3) {
        Serial.println("[MANUAL] Left");
        setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, false, 160, invertLeftMotor);
        setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, true, 160, invertRightMotor);
    } else if (command == 4) {
        Serial.println("[MANUAL] Right");
        setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, true, 160, invertLeftMotor);
        setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, false, 160, invertRightMotor);
    }
    currentAngularVelocity = 0.0;
}

void enqueueTimedCommand(String command, double startTime, double duration) {
    int nextTail = (queueTail + 1) % 16;
    if (nextTail == queueHead) {
        Serial.println("[MANUAL] Command queue full; command rejected");
        return;
    }
    commandQueue[queueTail].command = command;
    commandQueue[queueTail].startTime = startTime;
    commandQueue[queueTail].duration = duration;
    queueTail = nextTail;
    Serial.print("[MANUAL] Queued "); Serial.print(command);
    Serial.print(" for "); Serial.print(duration, 3); Serial.print(" seconds at ");
    Serial.println(startTime, 4);
}

int commandNumber(String command) {
    if (command == "FORWARD") return 1;
    if (command == "BACKWARD") return 2;
    if (command == "LEFT") return 3;
    if (command == "RIGHT") return 4;
    return 0;
}

void runTimedCommands() {
    double now = getSyncedTime();
    if (timedCommandActive) {
        if (activeTimedCommand == 1 && millis() - lastObstacleCheck > 100) {
            lastObstacleCheck = millis();
            if (obstacleTooClose()) {
                Serial.println("[SAFETY] Obstacle interrupted timed forward command");
                stopMotors(); timedCommandActive = false; activeTimedCommand = 0;
            }
        }
        if (timedCommandActive && now >= activeTimedEnd) {
            stopMotors(); timedCommandActive = false; activeTimedCommand = 0;
            Serial.println("[MANUAL] Timed command complete");
        }
    }
    if (!timedCommandActive && queueHead != queueTail && now >= commandQueue[queueHead].startTime) {
        activeTimedCommand = commandNumber(commandQueue[queueHead].command);
        activeTimedEnd = commandQueue[queueHead].startTime + commandQueue[queueHead].duration;
        queueHead = (queueHead + 1) % 16;
        driveManual(activeTimedCommand);
        if (activeTimedCommand > 0) timedCommandActive = true;
    }
}

void setMotor(int in1, int in2, int enablePin, bool forward, int speed, bool inverted) {
    bool actualForward = inverted ? !forward : forward;
    digitalWrite(in1, actualForward ? HIGH : LOW);
    digitalWrite(in2, actualForward ? LOW : HIGH);
    int scaledSpeed = constrain((speed * motorSpeedPercent) / 100, 0, 255);
    analogWrite(enablePin, scaledSpeed);
}

float readDistanceCm() {
    digitalWrite(ULTRASONIC_TRIG, LOW); delayMicroseconds(2);
    digitalWrite(ULTRASONIC_TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(ULTRASONIC_TRIG, LOW);
    unsigned long duration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000);
    if (duration == 0) return -1.0;
    return duration * 0.0343 / 2.0;
}

void sendStatus(String status) {
    Udp.beginPacket(laptopIP, laptopSyncPort);
    Udp.print(status);
    Udp.endPacket();
}

bool obstacleTooClose() {
    float distance = readDistanceCm();
    if (distance > 0) {
        Serial.print("[ULTRASONIC] Distance cm: "); Serial.println(distance, 1);
        sendStatus(String("ULTRA_STATUS,") + String(distance, 1));
    }
    return distance > 0 && distance <= OBSTACLE_STOP_DISTANCE_CM;
}

void calibrateForwardSpeed() {
    Serial.println("[CALIBRATION] Forward-speed calibration starting");
    sendStatus("CAL_STATUS,Running");
    float startDistance = readDistanceCm();
    if (startDistance <= 0 || startDistance < OBSTACLE_STOP_DISTANCE_CM + 30.0) {
        Serial.println("[CALIBRATION] Aborted: invalid or unsafe starting distance");
        sendStatus("CAL_STATUS,Aborted");
        stopMotors(); return;
    }
    const int calibrationPwm = 120;
    const unsigned long durationMs = 750;
    unsigned long started = millis();
    setMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, true, calibrationPwm, invertLeftMotor);
    setMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, true, calibrationPwm, invertRightMotor);
    delay(durationMs);
    stopMotors();
    float endDistance = readDistanceCm();
    float elapsedSeconds = (millis() - started) / 1000.0;
    if (endDistance > 0 && endDistance < startDistance && elapsedSeconds > 0) {
        calibratedForwardSpeedCmS = (startDistance - endDistance) / elapsedSeconds;
        Serial.print("[CALIBRATION] Forward speed cm/s: "); Serial.println(calibratedForwardSpeedCmS, 2);
        Serial.println("[CALIBRATION] Complete");
        sendStatus(String("CAL_STATUS,Complete ") + String(calibratedForwardSpeedCmS, 2) + " cm/s");
    } else {
        Serial.println("[CALIBRATION] Failed: no reliable distance change");
        sendStatus("CAL_STATUS,Failed");
    }
}

void stopMotors() {
    digitalWrite(LEFT_IN1, LOW); digitalWrite(LEFT_IN2, LOW);
    digitalWrite(RIGHT_IN1, LOW); digitalWrite(RIGHT_IN2, LOW);
    analogWrite(LEFT_EN, 0); analogWrite(RIGHT_EN, 0);
}
"""
    
    # Inject variables safely
    ino_code = ino_code.replace('__SSID__', ssid)
    ino_code = ino_code.replace('__PASS__', password)
    ino_code = ino_code.replace('__LAPTOP_IP_COMMAS__', laptop_ip.replace('.', ','))
    ino_code = ino_code.replace('__INVERT_LEFT_MOTOR__', 'true' if invert_left_motor else 'false')
    ino_code = ino_code.replace('__INVERT_RIGHT_MOTOR__', 'true' if invert_right_motor else 'false')
    ino_code = ino_code.replace('__ULTRASONIC_TRIG_PIN__', str(ultrasonic_trig_pin))
    ino_code = ino_code.replace('__ULTRASONIC_ECHO_PIN__', str(ultrasonic_echo_pin))
    ino_code = ino_code.replace('__OBSTACLE_STOP_DISTANCE_CM__', str(obstacle_stop_distance_cm))
    ino_code = ino_code.replace('__MOTOR_SPEED_PERCENT__', str(motor_speed_percent))
    ino_code = ino_code.replace('__TRACKING_DURATION_PERCENT__', str(tracking_duration_percent))
    for key, pin in motor_pins.items():
        ino_code = ino_code.replace(f'__{key.upper()}__', str(pin))

    return Response(ino_code, mimetype="text/plain", headers={"Content-disposition": "attachment; filename=robot_tracker.ino"})



if __name__ == '__main__':
    print("Dashboard Active at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)