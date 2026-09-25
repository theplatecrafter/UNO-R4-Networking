#include <WiFiS3.h>
#include <WiFiUdp.h>

const char ssid[] = "__SSID__";
const char pass[] = "__PASS__";
IPAddress laptopIP(__LAPTOP_IP_COMMAS__);
const unsigned int localPort = 8888;
const unsigned int telemetryPort = 8889;
const int LEFT_IN1 = __LEFT_MOTOR_IN1__;
const int LEFT_IN2 = __LEFT_MOTOR_IN2__;
const int LEFT_EN = __LEFT_MOTOR_EN__;
const int RIGHT_IN1 = __RIGHT_MOTOR_IN1__;
const int RIGHT_IN2 = __RIGHT_MOTOR_IN2__;
const int RIGHT_EN = __RIGHT_MOTOR_EN__;
const int ULTRASONIC_TRIG = __ULTRASONIC_TRIG_PIN__;
const int ULTRASONIC_ECHO = __ULTRASONIC_ECHO_PIN__;
const float OBSTACLE_STOP_DISTANCE_CM = __OBSTACLE_STOP_DISTANCE_CM__;

const float ACCELERATION_PERCENT_PER_SEC = 140.0;
const float DECELERATION_PERCENT_PER_SEC = 220.0;
const unsigned long CONTROL_UPDATE_INTERVAL_MS = 10;
const unsigned long OBSTACLE_CHECK_INTERVAL_MS = 60;
const unsigned long COMMAND_TIMEOUT_MS = 600;

WiFiUDP Udp;
char packetBuffer[64];
int targetLeftPercent = 0;
int targetRightPercent = 0;
float currentLeftPercent = 0.0;
float currentRightPercent = 0.0;
unsigned long lastCommandMs = 0;
unsigned long lastControlUpdateMs = 0;
unsigned long lastObstacleCheckMs = 0;
bool commandReceived = false;
bool obstacleLatched = false;
bool safetyStopActive = false;

void waitForNetwork();
bool hasValidLocalIp(const IPAddress& ip);
void receiveMotorCommand(const String& packet);
void updateSafetySensor();
float readDistanceCm();
void updateMotorRamp();
float approachMotorTarget(float current, float target, float step);
void applyMotors(int leftPercent, int rightPercent);
void applyMotor(int in1, int in2, int enablePin, int percent);
void stopForSafety();
void sendDistanceTelemetry(float distance);

void setup() {
  Serial.begin(115200);
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(LEFT_EN, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  pinMode(RIGHT_EN, OUTPUT);
  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);
  applyMotors(0, 0);

  Serial.println("[BOOT] Continuous motor controller starting");
  waitForNetwork();
  Udp.begin(localPort);
  Serial.print("[UDP] Listening on port ");
  Serial.println(localPort);
  Serial.println("[UDP] Protocol: MOTOR,leftPercent,rightPercent or STOP");
}

void loop() {
  int packetSize = Udp.parsePacket();
  if (packetSize > 0) {
    int length = Udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (length > 0) {
      packetBuffer[length] = '\0';
      receiveMotorCommand(String(packetBuffer));
    }
  }

  updateSafetySensor();
  if (commandReceived && millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    targetLeftPercent = 0;
    targetRightPercent = 0;
    commandReceived = false;
    Serial.println("[WATCHDOG] No motor target received; ramping to stop");
  }

  if (obstacleLatched) stopForSafety();
  else updateMotorRamp();
}

void waitForNetwork() {
  Serial.print("[WIFI] Connecting to SSID: ");
  Serial.println(ssid);
  WiFi.begin(ssid, pass);
  unsigned long attemptStarted = millis();
  unsigned long lastStatusPrint = 0;
  while (true) {
    if (WiFi.status() == WL_CONNECTED) {
      IPAddress ip = WiFi.localIP();
      if (hasValidLocalIp(ip)) {
        Serial.print("[WIFI] Connected; assigned IP: ");
        Serial.println(ip);
        return;
      }
    }
    if (millis() - lastStatusPrint >= 2000) {
      Serial.print("[WIFI] Waiting for hotspot/DHCP; status: ");
      Serial.print(WiFi.status());
      Serial.print(", IP: ");
      Serial.println(WiFi.localIP());
      lastStatusPrint = millis();
    }
    if (millis() - attemptStarted >= 15000) {
      Serial.println("[WIFI] No usable address; reconnecting");
      WiFi.disconnect();
      delay(1000);
      WiFi.begin(ssid, pass);
      attemptStarted = millis();
      lastStatusPrint = 0;
    }
    delay(250);
  }
}

bool hasValidLocalIp(const IPAddress& ip) {
  return ip[0] != 0 || ip[1] != 0 || ip[2] != 0 || ip[3] != 0;
}

void receiveMotorCommand(const String& packet) {
  if (packet == "STOP") {
    targetLeftPercent = 0;
    targetRightPercent = 0;
    lastCommandMs = millis();
    commandReceived = true;
    Serial.println("[MOTOR] STOP target received");
    return;
  }
  if (!packet.startsWith("MOTOR,")) {
    Serial.println("[UDP] Ignored packet; expected MOTOR,left,right or STOP");
    return;
  }
  int firstComma = packet.indexOf(',');
  int secondComma = packet.indexOf(',', firstComma + 1);
  if (firstComma < 0 || secondComma < 0 || packet.indexOf(',', secondComma + 1) >= 0) {
    Serial.println("[UDP] Invalid packet; use MOTOR,left,right");
    return;
  }

  targetLeftPercent = constrain(packet.substring(firstComma + 1, secondComma).toInt(), -100, 100);
  targetRightPercent = constrain(packet.substring(secondComma + 1).toInt(), -100, 100);
  lastCommandMs = millis();
  commandReceived = true;
  Serial.print("[MOTOR] Target left=");
  Serial.print(targetLeftPercent);
  Serial.print("%, right=");
  Serial.print(targetRightPercent);
  Serial.println("%");
}

void updateSafetySensor() {
  if (millis() - lastObstacleCheckMs < OBSTACLE_CHECK_INTERVAL_MS) return;
  lastObstacleCheckMs = millis();
  float distance = readDistanceCm();
  if (distance <= 0) return;

  obstacleLatched = distance <= OBSTACLE_STOP_DISTANCE_CM;
  sendDistanceTelemetry(distance);
  if (!obstacleLatched && safetyStopActive) {
    safetyStopActive = false;
    Serial.println("[SAFETY] Obstacle clear; resuming latest target");
  }
}

float readDistanceCm() {
  digitalWrite(ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG, LOW);
  unsigned long pulseDuration = pulseIn(ULTRASONIC_ECHO, HIGH, 30000);
  if (pulseDuration == 0) return -1.0;
  return pulseDuration * 0.0343 / 2.0;
}

void updateMotorRamp() {
  unsigned long nowMs = millis();
  if (nowMs - lastControlUpdateMs < CONTROL_UPDATE_INTERVAL_MS) return;
  float dt = (nowMs - lastControlUpdateMs) / 1000.0;
  lastControlUpdateMs = nowMs;
  if (dt <= 0.0) return;

  float leftRate = abs(targetLeftPercent) < abs(currentLeftPercent)
      || (currentLeftPercent != 0 && targetLeftPercent != 0 && ((currentLeftPercent < 0) != (targetLeftPercent < 0)))
      ? DECELERATION_PERCENT_PER_SEC : ACCELERATION_PERCENT_PER_SEC;
  float rightRate = abs(targetRightPercent) < abs(currentRightPercent)
      || (currentRightPercent != 0 && targetRightPercent != 0 && ((currentRightPercent < 0) != (targetRightPercent < 0)))
      ? DECELERATION_PERCENT_PER_SEC : ACCELERATION_PERCENT_PER_SEC;
  currentLeftPercent = approachMotorTarget(currentLeftPercent, targetLeftPercent, leftRate * dt);
  currentRightPercent = approachMotorTarget(currentRightPercent, targetRightPercent, rightRate * dt);
  applyMotors((int)round(currentLeftPercent), (int)round(currentRightPercent));
}

float approachMotorTarget(float current, float target, float step) {
  if (current != 0 && target != 0 && ((current < 0) != (target < 0))) target = 0;
  if (current < target) return min(current + step, target);
  if (current > target) return max(current - step, target);
  return current;
}

void applyMotors(int leftPercent, int rightPercent) {
  applyMotor(LEFT_IN1, LEFT_IN2, LEFT_EN, leftPercent);
  applyMotor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN, rightPercent);
}

void applyMotor(int in1, int in2, int enablePin, int percent) {
  percent = constrain(percent, -100, 100);
  if (percent == 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(enablePin, 0);
    return;
  }
  bool forward = percent > 0;
  digitalWrite(in1, forward ? HIGH : LOW);
  digitalWrite(in2, forward ? LOW : HIGH);
  analogWrite(enablePin, abs(percent) * 255 / 100);
}

void stopForSafety() {
  if (!safetyStopActive) Serial.println("[SAFETY] HC-SR04 stop; motor output disabled");
  safetyStopActive = true;
  currentLeftPercent = 0;
  currentRightPercent = 0;
  targetLeftPercent = 0;
  targetRightPercent = 0;
  applyMotors(0, 0);
}

void sendDistanceTelemetry(float distance) {
  Udp.beginPacket(laptopIP, telemetryPort);
  Udp.print("ULTRA_STATUS,");
  Udp.print(distance, 1);
  Udp.endPacket();
}