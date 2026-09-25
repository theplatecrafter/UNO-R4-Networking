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
    "pid_kp": 0.7,
    "pid_ki": 0.1,
    "pid_kd": 0.05,
    "forward_base_percent": 35,
    "turn_limit_percent": 55,
    "center_deadband_percent": 3.0,
    "manual_override": False,
    "manual_command": "STOP",
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
    "arduino_last_sent_at": "Never",
    "motor_inversion": {"left": False, "right": False},
    "manual_override": False,
    "manual_command": "STOP",
    "obstacle_distance_cm": -1.0,
    "tracking_enabled": True,
    "motor_speed_percent": 100,
    "pid_output": 0.0
}

runtime_logs = deque(maxlen=200)
runtime_log_lock = threading.Lock()

def add_log(message, level="INFO"):
    entry = {"time": time.strftime("%H:%M:%S"), "level": level, "message": message}
    with runtime_log_lock:
        runtime_logs.append(entry)
    getattr(logging.getLogger("robot_tracker"), level.lower(), logging.info)(message)


def build_motor_instruction(left_percent, right_percent):
    """Apply inversion and motor-speed tuning before serializing a target command."""
    with config_lock:
        speed_scale = CONFIG["motor_speed_percent"] / 100.0
        invert_left = CONFIG["invert_left_motor"]
        invert_right = CONFIG["invert_right_motor"]

    left = round(left_percent * speed_scale * (-1 if invert_left else 1))
    right = round(right_percent * speed_scale * (-1 if invert_right else 1))
    left = max(-100, min(100, left))
    right = max(-100, min(100, right))
    return f"MOTOR,{left},{right}", left, right


def send_motor_stop():
    with config_lock:
        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
    udp_out.sendto(b"STOP", arduino_address)
    telemetry["arduino_last_command"] = "STOP"
    telemetry["arduino_command_meaning"] = "Stop both motors immediately"
    telemetry["arduino_commands_sent"] += 1
    telemetry["arduino_last_sent_at"] = time.strftime("%H:%M:%S")


class PIDController:
    def __init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def update(self, error, kp, ki, kd, now):
        first_sample = self.previous_time is None
        dt = 0.02 if first_sample else max(now - self.previous_time, 1e-4)
        reset_derivative = first_sample
        if dt > 2.0:
            self.integral = 0.0
            reset_derivative = True
        self.previous_time = now
        self.integral = float(np.clip(self.integral + error * dt, -1.0, 1.0))
        derivative = 0.0 if reset_derivative else (error - self.previous_error) / dt
        self.previous_error = error
        return float(np.clip(kp * error + ki * self.integral + kd * derivative, -1.0, 1.0))

latest_raw_frame = None
latest_processed_frame = None
frame_lock = threading.Lock()

# --- UDP SENSOR TELEMETRY SERVER ---
def udp_telemetry_listener():
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
                add_log(f"Sensor telemetry listener bound to UDP port {bound_port}")
            data, addr = sync_sock.recvfrom(1024)
            message = data.decode().strip()
            if message.startswith("ULTRA_STATUS,"):
                try:
                    telemetry["obstacle_distance_cm"] = float(message.split(",", 1)[1])
                except ValueError:
                    pass
        except socket.timeout:
            continue
        except OSError as exc:
            add_log(f"Sensor telemetry listener error: {exc}", "ERROR")
            if sync_sock is not None:
                sync_sock.close()
            sync_sock = None
            time.sleep(1)
        except Exception:
            pass

threading.Thread(target=udp_telemetry_listener, daemon=True).start()

# --- VISION PROCESSING ENGINE ---
def process_video():
    global latest_raw_frame, latest_processed_frame, telemetry
    cap = None
    prev_time = time.time()
    fps_smooth = 0.0
    failed_reads = 0
    tracking_command_active = False
    filtered_target_angle = None
    pid = PIDController()
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
                tracking_enabled = CONFIG["tracking_enabled"]
                pid_kp = CONFIG["pid_kp"]
                pid_ki = CONFIG["pid_ki"]
                pid_kd = CONFIG["pid_kd"]
                forward_base_percent = CONFIG["forward_base_percent"]
                turn_limit_percent = CONFIG["turn_limit_percent"]
                center_deadband_percent = CONFIG["center_deadband_percent"]
            telemetry["tracking_enabled"] = tracking_enabled
            payload = None
            target_found = False
            if manual_override:
                pid.reset()
                processed_frame = raw_frame.copy()
                telemetry["status"] = f"Remote control: {manual_command}"
                telemetry["manual_override"] = True
                telemetry["manual_command"] = manual_command
            elif not tracking_enabled:
                pid.reset()
                processed_frame = raw_frame.copy()
                telemetry["status"] = "Tracking disabled"
                telemetry["tracking_enabled"] = False
            else:
                with tracker_lock:
                    processed_frame, target_found, x, y, radius = current_tracker.process_frame(raw_frame)

                if target_found:
                    offset_x = x - center_x
                    angle_deg = (offset_x / center_x) * (CONFIG["camera_fov_deg"] / 2.0)
                    if filtered_target_angle is None:
                        filtered_target_angle = angle_deg
                    else:
                        filtered_target_angle = 0.25 * angle_deg + 0.75 * filtered_target_angle
                    cv2.line(processed_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
                    telemetry["status"] = "Tracking"
                    telemetry["target_angle"] = round(filtered_target_angle, 1)

                    normalized_error = filtered_target_angle / (CONFIG["camera_fov_deg"] / 2.0)
                    if abs(normalized_error) <= center_deadband_percent / 100.0:
                        normalized_error = 0.0
                    pid_output = pid.update(normalized_error, pid_kp, pid_ki, pid_kd, time.monotonic())
                    turn_correction = pid_output * turn_limit_percent
                    left_percent = round(forward_base_percent + turn_correction)
                    right_percent = round(forward_base_percent - turn_correction)
                    left_percent = max(-100, min(100, left_percent))
                    right_percent = max(-100, min(100, right_percent))
                    telemetry["pid_output"] = round(pid_output, 3)
                    payload, final_left, final_right = build_motor_instruction(left_percent, right_percent)
                    tracking_command_active = final_left != 0 or final_right != 0
                else:
                    telemetry["status"] = "Searching"
                    filtered_target_angle = None
                    pid.reset()
                    telemetry["pid_output"] = 0.0
                    if tracking_command_active:
                        payload, _, _ = build_motor_instruction(0, 0)
                        tracking_command_active = False
                telemetry["manual_override"] = False
                telemetry["tracking_enabled"] = True

            if manual_override:
                command_meaning = f"Manual override: {manual_command}"
            elif not tracking_enabled:
                command_meaning = "Tracking disabled; camera-only mode"
            elif target_found:
                command_meaning = (
                    f"Tracking target at {telemetry['target_angle']} degrees; "
                    f"PID correction {telemetry['pid_output']:.3f}"
                )
            else:
                command_meaning = "No target detected; stop motors"
            telemetry["arduino_command_meaning"] = command_meaning

            # Tracking and safety commands are sent per frame; timed manual commands
            # are sent once by the manual-control endpoint.
            if payload is not None:
                with config_lock:
                    try:
                        arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
                        udp_out.sendto(payload.encode(), arduino_address)
                        telemetry["arduino_last_command"] = payload
                        telemetry["arduino_command_meaning"] = command_meaning
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
            pid_kp = float(data.get("pid_kp", CONFIG["pid_kp"]))
            pid_ki = float(data.get("pid_ki", CONFIG["pid_ki"]))
            pid_kd = float(data.get("pid_kd", CONFIG["pid_kd"]))
            forward_base_percent = int(data.get("forward_base_percent", CONFIG["forward_base_percent"]))
            turn_limit_percent = int(data.get("turn_limit_percent", CONFIG["turn_limit_percent"]))
            center_deadband_percent = float(data.get("center_deadband_percent", CONFIG["center_deadband_percent"]))
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
            if not all(0 <= gain <= 10 for gain in (pid_kp, pid_ki, pid_kd)):
                raise ValueError("PID gains must be between 0 and 10")
            if not 0 <= forward_base_percent <= 100 or not 0 <= turn_limit_percent <= 100:
                raise ValueError("Forward base and turn limit must be between 0 and 100 percent")
            if not 0 <= center_deadband_percent <= 25:
                raise ValueError("Center deadband must be between 0 and 25 percent of frame width")
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
                CONFIG["pid_kp"] = pid_kp
                CONFIG["pid_ki"] = pid_ki
                CONFIG["pid_kd"] = pid_kd
                CONFIG["forward_base_percent"] = forward_base_percent
                CONFIG["turn_limit_percent"] = turn_limit_percent
                CONFIG["center_deadband_percent"] = center_deadband_percent
                telemetry["arduino_status"] = f"Configured for {arduino_ip}:{arduino_port}"
                telemetry["motor_inversion"] = {"left": invert_left_motor, "right": invert_right_motor}
                telemetry["motor_speed_percent"] = motor_speed_percent
                telemetry["pid_output"] = 0.0
        except (TypeError, ValueError) as exc:
            add_log(f"Configuration rejected: {exc}", "ERROR")
            return jsonify({"status": "error", "message": str(exc)}), 400
        add_log(f"Configuration saved. Camera: {CONFIG['phone_url']}; Arduino: {CONFIG['arduino_ip']}:{CONFIG['arduino_port']}")
        add_log("Motor inversion and tuning settings will be applied to outgoing instructions")
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
    else:
        send_motor_stop()
    telemetry["status"] = "Starting engine..." if action == "start" else "Engine Stopped"
    add_log(f"Engine command: {action}")
    return jsonify({"status": "success"})

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
    telemetry["motor_inversion"] = {"left": left_inverted, "right": right_inverted}
    add_log(f"Motor inversion updated: left={left_inverted}, right={right_inverted}")
    return jsonify({"status": "success", "left": left_inverted, "right": right_inverted})

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
    telemetry["motor_speed_percent"] = speed_percent
    add_log(f"Motor speed set to {speed_percent}% for future instructions")
    return jsonify({"status": "success", "percent": speed_percent})

@app.route('/api/manual_mode', methods=['POST'])
def set_manual_mode():
    enabled = bool((request.get_json(silent=True) or {}).get("enabled", False))
    with config_lock:
        CONFIG["manual_override"] = enabled
        CONFIG["manual_command"] = "STOP"
    telemetry["manual_override"] = enabled
    telemetry["manual_command"] = "STOP"
    try:
        send_motor_stop()
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
    telemetry["tracking_enabled"] = enabled
    if not enabled:
        try:
            send_motor_stop()
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
    with config_lock:
        if not CONFIG["manual_override"]:
            return jsonify({"status": "error", "message": "Manual override is disabled"}), 409
        CONFIG["manual_command"] = command
    telemetry["manual_command"] = command
    directions = {
        "FORWARD": (100, 100),
        "BACKWARD": (-100, -100),
        "LEFT": (-100, 100),
        "RIGHT": (100, -100),
        "STOP": (0, 0),
    }
    try:
        if command == "STOP":
            packet = "STOP"
            final_left, final_right = 0, 0
        else:
            packet, final_left, final_right = build_motor_instruction(*directions[command])
        with config_lock:
            arduino_address = (CONFIG["arduino_ip"], CONFIG["arduino_port"])
        udp_out.sendto(packet.encode(), arduino_address)
        add_log(f"Queued manual command: {packet}")
        telemetry["arduino_last_command"] = packet
        return jsonify({"status": "success", "command": command, "packet": packet,
                "left_percent": final_left, "right_percent": final_right})
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
    pin_defaults = {
        "left_motor_in1": 2, "left_motor_in2": 3, "left_motor_en": 6,
        "right_motor_in1": 4, "right_motor_in2": 5, "right_motor_en": 9,
        "ultrasonic_trig_pin": 10, "ultrasonic_echo_pin": 11,
    }
    try:
        pins = {key: int(request.args.get(key, value)) for key, value in pin_defaults.items()}
        stop_distance = float(request.args.get('obstacle_stop_distance_cm', 20.0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid pin or stop-distance value"}), 400
    if any(not 0 <= pin <= 53 for pin in pins.values()):
        return jsonify({"status": "error", "message": "Arduino pins must be from 0 to 53"}), 400
    if len(set(pins.values())) != len(pins) or stop_distance <= 0:
        return jsonify({"status": "error", "message": "Pins must be unique and stop distance must be positive"}), 400

    sketch_path = os.path.join(os.path.dirname(__file__), "arduino", "robot_tracker.ino")
    with open(sketch_path, encoding="utf-8") as sketch_file:
        ino_code = sketch_file.read()
    replacements = {
        "__SSID__": ssid.replace('\\', '\\\\').replace('"', '\\"'),
        "__PASS__": password.replace('\\', '\\\\').replace('"', '\\"'),
        "__LAPTOP_IP_COMMAS__": laptop_ip.replace('.', ','),
        "__LEFT_MOTOR_IN1__": str(pins["left_motor_in1"]),
        "__LEFT_MOTOR_IN2__": str(pins["left_motor_in2"]),
        "__LEFT_MOTOR_EN__": str(pins["left_motor_en"]),
        "__RIGHT_MOTOR_IN1__": str(pins["right_motor_in1"]),
        "__RIGHT_MOTOR_IN2__": str(pins["right_motor_in2"]),
        "__RIGHT_MOTOR_EN__": str(pins["right_motor_en"]),
        "__ULTRASONIC_TRIG_PIN__": str(pins["ultrasonic_trig_pin"]),
        "__ULTRASONIC_ECHO_PIN__": str(pins["ultrasonic_echo_pin"]),
        "__OBSTACLE_STOP_DISTANCE_CM__": str(stop_distance),
    }
    for placeholder, value in replacements.items():
        ino_code = ino_code.replace(placeholder, value)
    return Response(ino_code, mimetype="text/plain", headers={"Content-disposition": "attachment; filename=robot_tracker.ino"})



if __name__ == '__main__':
    print("Dashboard Active at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)